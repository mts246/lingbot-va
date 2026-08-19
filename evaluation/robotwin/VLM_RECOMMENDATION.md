# VLM（视觉语言模型）推荐指南

## 最佳推荐（2024年性价比和性能综合考虑）

### 🥇 第一推荐：**Qwen-VL-Max (通义千问视觉模型)**

**推荐理由：**
- ✅ **强大的视觉理解能力**：在机器人任务场景表现优异
- ✅ **中文支持优秀**：阿里达摩院出品，对中文场景理解更好
- ✅ **性价比高**：价格相对便宜，API 稳定
- ✅ **低延迟**：国内访问速度快
- ✅ **支持多图像输入**：可以同时处理多个相机视角

**使用配置：**
```yaml
vlm:
  provider: "qwen"
  api_key: "YOUR_DASHSCOPE_API_KEY"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-vl-max"
```

**获取API Key：**
1. 访问：https://dashscope.aliyun.com/
2. 注册/登录阿里云账号
3. 开通 DashScope 服务
4. 获取 API Key

**定价：** 约 ¥0.008/千tokens（图像按tokens计算）

---

### 🥈 第二推荐：**GPT-4o (OpenAI)**

**推荐理由：**
- ✅ **最强性能**：视觉理解能力业界顶尖
- ✅ **推理准确度高**：适合复杂场景判断
- ✅ **文档和社区支持完善**
- ⚠️ **价格较高**：成本是Qwen的3-5倍
- ⚠️ **需要稳定的国际网络**

**使用配置：**
```yaml
vlm:
  provider: "openai"
  api_key: "YOUR_OPENAI_API_KEY"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
```

**定价：** $0.005/图像（标准分辨率）

---

### 🥉 第三推荐：**GLM-4V (智谱清言)**

**推荐理由：**
- ✅ **国产开源**：清华系出品，支持私有化部署
- ✅ **价格亲民**：API 价格便宜
- ✅ **中文场景优化**
- ⚠️ **性能稍逊于 Qwen 和 GPT-4**

**使用配置：**
```yaml
vlm:
  provider: "zhipu"
  api_key: "YOUR_ZHIPU_API_KEY"
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  model: "glm-4v"
```

---

### 🤔 第四选择：**DeepSeek-VL**

**注意事项：**
- ⚠️ **当前 DeepSeek 的 VL 能力可能还在内测中**
- ✅ 如果你已经有 DeepSeek API Key，可以尝试
- ✅ 价格便宜（如果支持的话）

**使用配置：**
```yaml
vlm:
  provider: "deepseek"
  api_key: "YOUR_DEEPSEEK_API_KEY"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-vl"  # 需要确认是否支持
```

---

## 性能对比表

| 模型 | 视觉理解 | 推理能力 | 中文支持 | 价格 | 延迟 | 推荐指数 |
|------|---------|---------|---------|------|------|---------|
| Qwen-VL-Max | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ¥¥ | 快 | ⭐⭐⭐⭐⭐ |
| GPT-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ¥¥¥¥ | 中 | ⭐⭐⭐⭐ |
| GLM-4V | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ¥ | 快 | ⭐⭐⭐⭐ |
| DeepSeek-VL | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ¥ | 快 | ⭐⭐⭐ |

---

## 实际使用建议

### 对于你的场景（机器人操作任务判断）：

**最佳选择：Qwen-VL-Max**
```python
# 在 stage_config.yaml 中配置
vlm:
  provider: "qwen"
  api_key: "YOUR_DASHSCOPE_API_KEY"  # 或从环境变量读取
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-vl-max"
  temperature: 0.1
  max_tokens: 1000
```

**理由：**
1. **多视角支持好**：你的代码中有 3 个相机视角（cam_high, cam_left_wrist, cam_right_wrist），Qwen-VL 对多图像输入支持完善
2. **物体识别准确**：对"绿色瓶子"、"抓取"、"接触"等机器人任务场景理解准确
3. **性价比高**：大量测试不会产生过高成本
4. **延迟低**：实时性更好

---

## 快速测试脚本

创建 `test_vlm.py` 测试各个 VLM 的效果：

```python
import os
import numpy as np
from stage_manager import StageCompletionChecker
from PIL import Image

# 加载测试图像
test_obs = {
    "observation.images.cam_high": np.array(Image.open("test_high.jpg")),
    "observation.images.cam_left_wrist": np.array(Image.open("test_left.jpg")),
    "observation.images.cam_right_wrist": np.array(Image.open("test_right.jpg")),
}

# 测试阶段信息
stage_info = {
    "stage_id": 1,
    "prompt": "Move gripper above the green bottle",
    "description": "Navigate robot arm to position above the target green bottle",
    "success_criteria": "Gripper is positioned directly above the green bottle within 5cm"
}

# 测试 Qwen-VL
checker = StageCompletionChecker(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-vl-max"
)

is_completed, reasoning, confidence = checker.check_stage_completion(test_obs, stage_info)
print(f"Completed: {is_completed}, Confidence: {confidence}")
print(f"Reasoning: {reasoning}")
```

---

## 环境变量设置

在你的服务器上设置环境变量：

```bash
# 推荐：Qwen-VL
export DASHSCOPE_API_KEY="your_dashscope_api_key"

# 备选：OpenAI GPT-4o
export OPENAI_API_KEY="your_openai_api_key"

# 备选：智谱 GLM-4V
export ZHIPU_API_KEY="your_zhipu_api_key"

# DeepSeek (已有)
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

---

## 最终建议

**为你的项目，我强烈推荐使用 Qwen-VL-Max：**

1. **性能**：足够准确判断机器人任务场景
2. **成本**：价格合理，适合大规模测试
3. **速度**：国内访问快，延迟低
4. **兼容性**：支持多图像输入，完美匹配你的多相机设置

如果预算充足且需要最高准确度，可以考虑 GPT-4o 作为备选。
