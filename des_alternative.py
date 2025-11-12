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

# Load precomputed inputs (contains the processed image)
precomputed_inputs = torch.load("inputs.pt", weights_only=False).to(model.device)

# Option 2: Reuse image data but create new text prompt
new_messages = [
    [{"role": "user", "content": [{"type": "image", "image": "file:///workspace/ai-create-embeddings/fordcasepage3.png"}, {"type": "text", "text": "Extract the text from the image."}]}],
]

# Create new text prompt while reusing the pixel_values from precomputed_inputs
new_text = processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)
text_only_inputs = processor(text=new_text, return_tensors="pt").to(model.device)

# Combine new text with original image data
modified_inputs = {
    "input_ids": text_only_inputs["input_ids"],
    "attention_mask": text_only_inputs["attention_mask"],
    "pixel_values": precomputed_inputs["pixel_values"],  # Reuse original image
    "image_grid_thw": precomputed_inputs["image_grid_thw"]
}

output = model.generate(**modified_inputs)
print(processor.batch_decode(output, skip_special_tokens=True))
