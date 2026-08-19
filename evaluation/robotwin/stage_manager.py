"""
Stage-based Task Execution Manager
Author: genghaotian
Description: 使用 LLM 进行任务分解，使用 VLM 进行阶段完成判断
"""

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Any
import requests
from PIL import Image
import numpy as np


class TaskDecomposer:
    """使用 LLM 将复杂任务分解为多个简单阶段"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        """
        初始化任务分解器
        
        Args:
            api_key: DeepSeek API Key
            base_url: API 基础 URL
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def decompose_task(self, complex_prompt: str, max_stages: int = 5) -> List[Dict[str, Any]]:
        """
        将复杂任务分解为多个阶段
        
        Args:
            complex_prompt: 复杂的任务描述
            max_stages: 最大阶段数
            
        Returns:
            List of stage dicts with keys: 'stage_id', 'prompt', 'description', 'success_criteria'
        """
        decompose_prompt = f"""You are a robot task planner for a dual-arm tabletop manipulation benchmark.
Your job is to split a complex instruction into a small number of VISUALLY CHECKABLE stages.

Complex Task: {complex_prompt}

Very important constraints:
1. Use at most {max_stages} stages. Prefer 2-3 coarse stages over many fine stages.
2. Each stage must be easy for a VLM to judge from RGB images only.
3. Do NOT use precise metric thresholds such as "within 5cm", "slightly", or "briefly" in success criteria.
4. Avoid stages whose completion requires detecting subtle contact, force, tiny gripper motion, or exact distance.
5. Prefer observable state changes:
   - object is lifted and no longer resting on the table
   - object is inside/on/next to a target container or area
   - door is visibly open
   - switch state visibly changed
   - gripper/arm has clearly moved away after an interaction
6. If the original task contains a hard-to-see micro-action such as "touch", merge it with a visually obvious proxy, e.g.
   "left arm reaches the object and then moves away" rather than "detect exact contact".
7. The stage prompt should still be useful as an action instruction for the policy model.
8. The success criteria should be phrased as robust visual evidence, not fine-grained geometry.

For each stage, provide:
- prompt: concise action instruction for the robot policy
- description: what this stage does
- success_criteria: what a VLM can reliably see after completion
- visual_evidence: 2-4 concrete visual cues the VLM should look for
- failure_evidence: 2-4 visual cues indicating the stage is not complete

Output format (JSON only):
{{
  "stages": [
    {{
      "stage_id": 1,
      "prompt": "Coarse action instruction for the robot",
      "description": "Detailed explanation of this stage",
      "success_criteria": "Robust observable condition that indicates completion",
      "visual_evidence": ["clear visual cue 1", "clear visual cue 2"],
      "failure_evidence": ["failure cue 1", "failure cue 2"]
    }}
  ]
}}

Example for "Use the left arm to briefly touch the green bottle, then use the right arm to pick up the same green bottle":
{{
  "stages": [
    {{
      "stage_id": 1,
      "prompt": "Use the left arm to reach the green bottle, touch or approach it, then move the left arm away",
      "description": "Complete the left-arm interaction with the green bottle without requiring exact contact detection.",
      "success_criteria": "The left arm has completed its interaction and is visibly no longer blocking the green bottle, while the green bottle remains identifiable on the table.",
      "visual_evidence": [
        "the left gripper is near or has passed the green bottle",
        "the left arm is no longer directly over the bottle",
        "the green bottle is still visible and stable"
      ],
      "failure_evidence": [
        "the left arm is still far from the bottle",
        "the left gripper is still hovering over or blocking the bottle",
        "the green bottle is not visible"
      ]
    }},
    {{
      "stage_id": 2,
      "prompt": "Use the right arm to grasp and lift the green bottle",
      "description": "Move the right gripper to the green bottle and lift it from the table.",
      "success_criteria": "The green bottle is visibly held by the right gripper or has clearly moved upward from its original table position.",
      "visual_evidence": [
        "right gripper is around or attached to the green bottle",
        "green bottle is no longer flat on its original table location",
        "the bottle moves together with the right arm"
      ],
      "failure_evidence": [
        "green bottle remains untouched on the table",
        "right gripper is far from the bottle",
        "the bottle has fallen or is not visible"
      ]
    }}
  ]
}}

Now decompose the given task and output ONLY the JSON:"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful robot task planning assistant. Always respond with valid JSON."},
                        {"role": "user", "content": decompose_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 尝试解析 JSON（处理可能的markdown代码块）
            content = content.strip()
            if content.startswith('```'):
                # 移除markdown代码块标记
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace('```json', '').replace('```', '').strip()
            
            parsed = json.loads(content)
            stages = parsed.get('stages', [])
            
            print(f"[TaskDecomposer] Successfully decomposed task into {len(stages)} stages")
            return stages
            
        except Exception as e:
            print(f"[TaskDecomposer] Error decomposing task: {e}")
            # 返回原始任务作为单一阶段
            return [{
                "stage_id": 1,
                "prompt": complex_prompt,
                "description": "Execute the full task",
                "success_criteria": "Task completed"
            }]


class StageCompletionChecker:
    """使用 VLM 判断当前阶段是否完成"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", 
                 model: str = "deepseek-vl", provider: str = "deepseek"):
        """
        初始化阶段完成检查器
        
        Args:
            api_key: API Key
            base_url: API 基础 URL
            model: 使用的模型名称
            provider: VLM 提供商 ('deepseek', 'qwen', 'openai')
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将 numpy 图像编码为 base64"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _to_uint8_image(self, image: np.ndarray) -> np.ndarray:
        """Convert an observation image to uint8 RGB for logging."""
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _save_check_artifacts(
        self,
        log_dir: Path,
        images_to_check: List[np.ndarray],
        image_descriptions: List[str],
        stage_info: Dict,
        check_prompt: str,
        raw_response: str,
        parsed_response: Optional[Dict[str, Any]],
        is_completed: bool,
        reasoning: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Save VLM check inputs and outputs for debugging and audit."""
        log_dir.mkdir(parents=True, exist_ok=True)
        image_files = []
        safe_names = {
            "Top-down view": "cam_high",
            "Left wrist view": "cam_left_wrist",
            "Right wrist view": "cam_right_wrist",
        }
        for idx, (img, desc) in enumerate(zip(images_to_check, image_descriptions)):
            image_name = f"{idx}_{safe_names.get(desc, desc.lower().replace(' ', '_'))}.jpg"
            image_path = log_dir / image_name
            Image.fromarray(self._to_uint8_image(img)).save(image_path, quality=95)
            image_files.append({"description": desc, "file": image_name})

        payload = {
            "metadata": metadata or {},
            "stage_info": stage_info,
            "check_prompt": check_prompt,
            "images": image_files,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "completed": is_completed,
            "reasoning": reasoning,
            "confidence": confidence,
        }
        with open(log_dir / "check_result.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    def check_stage_completion(self, obs: Dict, stage_info: Dict, 
                                history_images: Optional[List[np.ndarray]] = None,
                                log_dir: Optional[str | Path] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> tuple[bool, str, float]:
        """
        判断当前阶段是否完成
        
        Args:
            obs: 当前观察，包含多个相机视角的图像
            stage_info: 阶段信息字典，包含 'prompt', 'description', 'success_criteria'
            history_images: 可选的历史图像列表，用于对比
            
        Returns:
            (is_completed, reasoning, confidence)
        """
        # 准备图像
        images_to_check = []
        image_descriptions = []
        
        # 主要视角
        if "observation.images.cam_high" in obs:
            images_to_check.append(obs["observation.images.cam_high"])
            image_descriptions.append("Top-down view")
        
        if "observation.images.cam_left_wrist" in obs:
            images_to_check.append(obs["observation.images.cam_left_wrist"])
            image_descriptions.append("Left wrist view")
        
        if "observation.images.cam_right_wrist" in obs:
            images_to_check.append(obs["observation.images.cam_right_wrist"])
            image_descriptions.append("Right wrist view")
        
        if not images_to_check:
            print("[StageCompletionChecker] No images found in observation")
            return False, "No visual observation available", 0.0
        
        visual_evidence = stage_info.get("visual_evidence", [])
        failure_evidence = stage_info.get("failure_evidence", [])

        # 构造检查prompt
        check_prompt = f"""You are a conservative robot task stage verifier using only RGB images.
Your goal is not to judge the final task success, but only whether the CURRENT STAGE is ready to switch to the next stage.

Current Stage Goal: {stage_info['prompt']}
Stage Description: {stage_info['description']}
Success Criteria: {stage_info['success_criteria']}
Expected Visual Evidence: {json.dumps(visual_evidence, ensure_ascii=False)}
Failure Evidence: {json.dumps(failure_evidence, ensure_ascii=False)}

Available camera views: {', '.join(image_descriptions)}
- Top-down view is best for object locations and arm-object relationships.
- Wrist views are useful for whether a gripper is near, holding, or blocking an object.

Verification policy:
1. Use robust visual cues, not precise geometry. Do not estimate exact centimeters or exact contact force.
2. Treat "touch/contact" as complete only when there is a clear visual proxy: gripper reached the object area and then moved away, object pose changed, or object is no longer blocked by that arm.
3. For "move above/near" stages, require the relevant gripper to be clearly near the target object in at least one view; do not require exact alignment.
4. For "grasp/lift/pick" stages, prefer evidence that the object is held by the gripper or visibly displaced/lifted from its previous table position.
5. For "place/put" stages, prefer evidence that the object is resting at/on/inside the target area and the gripper is released or moving away.
6. For "open/press/switch" stages, prefer visible state change of the object/device rather than subtle arm contact.
7. If views conflict, trust the view where the target object and relevant gripper are most visible.
8. If the target object or relevant gripper is heavily occluded, set completed=false unless another clear cue proves completion.
9. Be conservative for stage switching: completed=true only when the stage is probably complete and continuing the same stage is less useful than moving to the next stage.
10. However, do not be overly strict about tiny distances, exact contact, or exact final pose if the coarse stage objective is visually achieved.

Return JSON only with this schema:
{{
  "completed": true/false,
  "confidence": 0.0,
  "reasoning": "short but specific explanation based on visible evidence",
  "best_view": "Top-down view / Left wrist view / Right wrist view / multiple",
  "matched_visual_evidence": ["which expected cues are visible"],
  "matched_failure_evidence": ["which failure cues are visible"],
  "occlusion_or_uncertainty": "what is unclear, if anything",
  "switch_recommendation": "switch / continue"
}}

Output ONLY valid JSON:"""

        try:
            # 编码图像
            image_contents = []
            for img in images_to_check:
                img_base64 = self._encode_image(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # 构造消息（图像 + 文本）
            messages = [
                {
                    "role": "user",
                    "content": [
                        *image_contents,
                        {"type": "text", "text": check_prompt}
                    ]
                }
            ]
            
            # 调用 VLM API
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 解析响应
            content = content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace('```json', '').replace('```', '').strip()
            
            parsed = json.loads(content)
            
            is_completed = parsed.get('completed', False)
            reasoning = parsed.get('reasoning', 'No reasoning provided')
            confidence = parsed.get('confidence', 0.5)

            if log_dir is not None:
                # codeflicker-fix: CHECK_LOGGING-Issue-001/r18bb54jk2csmy8m1dcz
                # Persist every VLM check input image, stage metadata, and model response.
                self._save_check_artifacts(
                    log_dir=Path(log_dir),
                    images_to_check=images_to_check,
                    image_descriptions=image_descriptions,
                    stage_info=stage_info,
                    check_prompt=check_prompt,
                    raw_response=result['choices'][0]['message']['content'],
                    parsed_response=parsed,
                    is_completed=is_completed,
                    reasoning=reasoning,
                    confidence=confidence,
                    metadata=metadata,
                )
            
            print(f"[StageCompletionChecker] Stage {stage_info['stage_id']}: "
                  f"{'✓ COMPLETED' if is_completed else '✗ NOT COMPLETED'} "
                  f"(confidence: {confidence:.2f})")
            print(f"  Reasoning: {reasoning[:100]}...")
            
            return is_completed, reasoning, confidence
            
        except Exception as e:
            print(f"[StageCompletionChecker] Error checking completion: {e}")
            if log_dir is not None:
                self._save_check_artifacts(
                    log_dir=Path(log_dir),
                    images_to_check=images_to_check,
                    image_descriptions=image_descriptions,
                    stage_info=stage_info,
                    check_prompt=check_prompt,
                    raw_response=f"Error during check: {str(e)}",
                    parsed_response=None,
                    is_completed=False,
                    reasoning=f"Error during check: {str(e)}",
                    confidence=0.0,
                    metadata=metadata,
                )
            # 发生错误时保守处理，假设未完成
            return False, f"Error during check: {str(e)}", 0.0


class StageManager:
    """管理多阶段任务执行流程"""
    
    def __init__(self, stages: List[Dict], completion_threshold: float = 0.7):
        """
        初始化阶段管理器
        
        Args:
            stages: 阶段列表
            completion_threshold: 阶段完成的最低置信度阈值
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.completion_threshold = completion_threshold
        self.stage_history = []
        
        print(f"[StageManager] Initialized with {len(stages)} stages")
        for i, stage in enumerate(stages):
            print(f"  Stage {i+1}: {stage['prompt']}")
    
    def get_current_stage(self) -> Optional[Dict]:
        """获取当前阶段信息"""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def get_current_prompt(self) -> str:
        """获取当前阶段的 prompt"""
        stage = self.get_current_stage()
        return stage['prompt'] if stage else ""
    
    def has_next_stage(self) -> bool:
        """是否还有下一个阶段"""
        return self.current_stage_idx < len(self.stages) - 1
    
    def advance_to_next_stage(self, reasoning: str = "", confidence: float = 1.0):
        """切换到下一阶段"""
        current_stage = self.get_current_stage()
        if current_stage:
            self.stage_history.append({
                "stage_id": current_stage['stage_id'],
                "prompt": current_stage['prompt'],
                "reasoning": reasoning,
                "confidence": confidence
            })
        
        self.current_stage_idx += 1
        next_stage = self.get_current_stage()
        
        if next_stage:
            print(f"\n[StageManager] ═══ Advancing to Stage {self.current_stage_idx + 1}/{len(self.stages)} ═══")
            print(f"  Goal: {next_stage['prompt']}")
            print(f"  Success Criteria: {next_stage['success_criteria']}")
        else:
            print(f"\n[StageManager] ✓ All {len(self.stages)} stages completed!")
    
    def is_all_completed(self) -> bool:
        """是否所有阶段都已完成"""
        return self.current_stage_idx >= len(self.stages)
    
    def get_progress_summary(self) -> Dict:
        """获取执行进度摘要"""
        return {
            "total_stages": len(self.stages),
            "current_stage": self.current_stage_idx + 1,
            "completed_stages": len(self.stage_history),
            "progress_percentage": (len(self.stage_history) / len(self.stages)) * 100 if self.stages else 0,
            "history": self.stage_history
        }


# 便捷工厂函数
def create_stage_based_system(
    llm_api_key: str,
    vlm_api_key: Optional[str] = None,
    llm_base_url: str = "https://api.deepseek.com/v1",
    vlm_base_url: Optional[str] = None,
    vlm_model: str = "deepseek-vl",
    completion_threshold: float = 0.7
) -> tuple[TaskDecomposer, StageCompletionChecker, callable]:
    """
    创建完整的阶段式执行系统
    
    Args:
        llm_api_key: LLM API Key (用于任务分解)
        vlm_api_key: VLM API Key (用于阶段判断)，如果为None则使用llm_api_key
        llm_base_url: LLM API 基础URL
        vlm_base_url: VLM API 基础URL，如果为None则使用llm_base_url
        vlm_model: VLM 模型名称
        completion_threshold: 完成判断的置信度阈值
        
    Returns:
        (task_decomposer, stage_checker, create_manager_fn)
    """
    vlm_api_key = vlm_api_key or llm_api_key
    vlm_base_url = vlm_base_url or llm_base_url
    
    task_decomposer = TaskDecomposer(
        api_key=llm_api_key,
        base_url=llm_base_url
    )
    
    stage_checker = StageCompletionChecker(
        api_key=vlm_api_key,
        base_url=vlm_base_url,
        model=vlm_model
    )
    
    def create_manager(complex_prompt: str, max_stages: int = 5) -> StageManager:
        stages = task_decomposer.decompose_task(complex_prompt, max_stages)
        return StageManager(stages, completion_threshold)
    
    return task_decomposer, stage_checker, create_manager
