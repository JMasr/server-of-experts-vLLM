import os
from pathlib import Path

from huggingface_hub import snapshot_download
from server_experts.config import logger


def download_hugging_face_model(model_path: Path, model_id: str, hf_token: str) -> bool:
    """
    Download a Hugging Face model to the specified local path.

    Args:
        model_path (Path): Destination directory to store the model.
        model_id (str): Repository id of the model to download.
        hf_token (str): Hugging Face API token for authentication.

    Returns:
        bool: True if the model was successfully downloaded, False otherwise.
    """
    os.makedirs(model_path, exist_ok=True)

    logger.info(f"Downloading {model_id} to {model_path}...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=model_path,
            token=hf_token
        )

        logger.info("Download complete!")
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_id}: {e}")
        return False