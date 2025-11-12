import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load model and processor
model_name = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen3VLForConditionalGeneration.from_pretrained(
  model_name,
  device_map="auto",
  torch_dtype=torch.float16
)

precomputed_features = torch.load("inputs.pt", weights_only=False).to(model.device)

print(precomputed_features)
print(type(precomputed_features))
print(precomputed_features.keys())