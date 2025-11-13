from huggingface_hub import hf_hub_download

def download_model(model_name: str, cache_dir: str = None) -> str:
    """
    Downloads the specified model from Hugging Face Hub.

    Args:
        model_name (str): The name of the model to download.
        cache_dir (str, optional): Directory to cache the downloaded model.

    Returns:
        str: Path to the downloaded model file.
    """
    try:
        model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", cache_dir=cache_dir)
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}") from e
    
