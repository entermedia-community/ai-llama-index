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

# Load precomputed visual features
precomputed_features = torch.load("inputs.pt", weights_only=False).to(model.device)

# Example: Inject features into the model’s processing
def generate_with_cached_features(prompt, visual_features):
    
    text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    inputs_embeds = model.get_input_embeddings()(text_inputs["input_ids"])
    
    combined_embeds = torch.cat([visual_features, inputs_embeds], dim=1)

    outputs = model.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=100
    )
    return outputs

prompt = "Describe this image."
output = generate_with_cached_features(prompt, precomputed_features['pixel_values'])
print(processor.decode(output[0], skip_special_tokens=True))