from typing import List
import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    """Simple wrapper around a Hugging Face transformer for embeddings.

    Inputs:
      - model_name: HF model name (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
    Methods:
      - embed(texts) -> List[List[float]]
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Mean pooling of last hidden state
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, dim)
        attention_mask = inputs.get("attention_mask")

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()
