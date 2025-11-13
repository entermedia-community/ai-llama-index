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

precomputed_inputs = torch.load("inputs-alt.pt", weights_only=False).to(model.device)
precomputed_inputs = precomputed_inputs.unsqueeze(0)

text_inputs = processor(text="What is this image about?", return_tensors="pt").to(model.device)

output = model.generate(**precomputed_inputs, **text_inputs)
decoded_output = processor.batch_decode(output, skip_special_tokens=True)

for i, text in enumerate(decoded_output):
    print(f"Output {i}:")
    print(text)
    print("-" * 80)

print(model.device)
