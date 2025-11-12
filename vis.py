import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen3-VL-8B-Instruct"
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    [{"role": "user", "content": [{"type": "image", "image": "file:///workspace/ai-create-embeddings/fordcasepage3.png"}, {"type": "text", "text": "Describe this image."}]}],
]

processor = AutoProcessor.from_pretrained(model_name)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype="auto", device_map="auto")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

images, _ = process_vision_info(messages, image_patch_size=16)

print(images)

inputs = processor(text=text, images=images, return_tensors="pt")
inputs = inputs.to(model.device)

torch.save(inputs, "inputs.pt")

print(inputs.keys())