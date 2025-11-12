import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

# Load model and processor
model_name = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

precomputed_inputs = torch.load("inputs.pt", weights_only=False).to(model.device)
text_inputs = processor(text="Extract the text from the image.", return_tensors="pt").to(model.device)

output = model.generate(**precomputed_inputs)
print(processor.batch_decode(output, skip_special_tokens=True))

print(list(precomputed_inputs.keys()))
print(list(text_inputs.keys()))