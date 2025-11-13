from typing import Dict
import os
import torch
import numpy as np
from PIL import Image
import logging

# Module logger
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


try:
    # Import Qwen3V model and processor (must be available in the environment)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except Exception as e:
    logger.error("Failed to import Qwen3V classes from transformers: %s: %s", type(e).__name__, e)
    raise ImportError("Qwen3V support is required: install a transformers build that provides Qwen3V") from e

# Default path can be overridden via model_name parameter
DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def extract_multimodal_embeddings(text: str, image_path: str, device: str = None, model_name: str = None) -> Dict[str, torch.Tensor]:
    """Return a dict with 'visual', 'text', 'multimodal', and 'input_ids'.
    
    Uses a local Qwen3V model to process text and image inputs together and extract 
    multimodal embeddings. The model should be a local file (typically .gguf or .bin) 
    or a directory containing the model files.
    
    Args:
        text: The text to embed
        image_path: Path to the image file to embed
        device: Device to run inference on ('cuda' or 'cpu')
        model_name: Full path to the local model. If not provided, uses DEFAULT_QWEN3_MODEL
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("extract_multimodal_embeddings: device=%s, model_name=%s", device, model_name)

    image = Image.open(image_path).convert("RGB")
    logger.debug("Opened image %s (size=%s)", image_path, image.size)

    # Use Qwen3V. Allow overriding model_name.

    model_path = model_name or DEFAULT_QWEN3_MODEL
    logger.info("Using Qwen3V model from path: %s", model_path)

    # if not os.path.exists(model_path):
    #     raise RuntimeError(f"Model path does not exist: {model_path}")

    logger.debug("Loading processor")
    processor = AutoProcessor.from_pretrained(model_path)
    
    logger.debug("Loading model from path: %s", model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto", 
        device_map="auto"
    ).to(device)
    logger.info("Model loaded from %s", model_path)

    # Process inputs
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Try high-level convenience methods if present
        visual_emb = None
        text_emb = None
        if hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
            try:
                visual_emb = model.get_image_features(pixel_values=inputs.get("pixel_values"))
                logger.debug("Qwen3V: extracted visual_emb shape=%s", getattr(visual_emb, 'shape', None))
            except Exception:
                logger.exception("Qwen3V: get_image_features failed")
                visual_emb = None
            try:
                text_emb = model.get_text_features(input_ids=inputs.get("input_ids"))
                logger.debug("Qwen3V: extracted text_emb shape=%s", getattr(text_emb, 'shape', None))
            except Exception:
                logger.exception("Qwen3V: get_text_features failed")
                text_emb = None

        # Last-resort: use encoder outputs if available
        enc_out = None
        if visual_emb is None or text_emb is None:
            try:
                enc = model.get_encoder()
                enc_out = enc(**inputs)
                if visual_emb is None:
                    visual_emb = getattr(enc_out, "image_embeds", None)
                if text_emb is None:
                    text_emb = getattr(enc_out, "text_embeds", None)
            except Exception:
                enc_out = None

        # Fallback to last_hidden_state mean pooling
        if visual_emb is None and enc_out is not None:
            visual_emb = getattr(enc_out, "last_hidden_state", None)
            if visual_emb is not None:
                visual_emb = visual_emb.mean(dim=1)
        if text_emb is None and enc_out is not None:
            text_emb = getattr(enc_out, "last_hidden_state", None)
            if text_emb is not None:
                text_emb = text_emb.mean(dim=1)

        if visual_emb is None or text_emb is None:
            logger.error("Qwen3V failed to extract features: visual=%s text=%s", visual_emb is not None, text_emb is not None)
            raise RuntimeError("Unable to extract image/text features from Qwen model instance; inspect model API in your transformers version.")

        multimodal = torch.cat([visual_emb, text_emb], dim=-1).squeeze()

    return {
        "visual": visual_emb.squeeze(),
        "text": text_emb.squeeze(),
        "multimodal": multimodal,
        "input_ids": inputs.get("input_ids") if "input_ids" in inputs else torch.tensor([])
    }

    # All code paths above either return or raise; keep this as a safety net
    raise RuntimeError("Unable to load Qwen3V model or extract embeddings.")


def save_visual_embeddings(text: str, image_path: str, output_path: str, device: str = None, model_name: str = None):
    embeddings = extract_multimodal_embeddings(text, image_path, device=device, model_name=model_name)

    torch.save({
        "visual_embedding": embeddings["visual"].cpu(),
        "text_embedding": embeddings["text"].cpu(),
        "multimodal_embedding": embeddings["multimodal"].cpu(),
        "input_ids": embeddings.get("input_ids", torch.tensor([])).cpu(),
        "text": text,
        "image_path": image_path,
    }, output_path)

    # Also save numpy archive
    np.savez(
        output_path.replace(".pt", ".npz"),
        visual=embeddings["visual"].cpu().numpy(),
        text=embeddings["text"].cpu().numpy(),
        multimodal=embeddings["multimodal"].cpu().numpy(),
    )

    logger.info("Saved embeddings to %s (.pt and .npz)", output_path)
    logger.debug("Embedding shapes: visual=%s text=%s multimodal=%s",
                 getattr(embeddings['visual'], 'shape', None),
                 getattr(embeddings['text'], 'shape', None),
                 getattr(embeddings['multimodal'], 'shape', None))
