# Credits: Hub integration utilities for training checkpoints.

"""Training utilities: checkpoint upload to Hugging Face Hub."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def push_checkpoint_to_hub(
    folder_path: str | Path,
    hub_model_id: str,
    commit_message: str,
    path_in_repo: str,
    request_timeout: int = 30,
    token: str | None = None,
) -> None:
    """Upload a checkpoint directory to Hugging Face Hub.

    Parameters
    ----------
    folder_path : str | Path
        Local path to the checkpoint directory (e.g. ``checkpoint-epoch-50``).
    hub_model_id : str
        Hugging Face repo id (e.g. ``"username/repo-name"``).
    commit_message : str
        Commit message for the upload.
    path_in_repo : str
        Destination path within the repo (e.g. ``"pix2pix/facades/checkpoint-epoch-50"``).
    request_timeout : int
        Timeout in seconds for Hub requests (default 30).
    token : str | None
        HF token. If None, uses ``HF_TOKEN`` env var or cached token.
    """
    path = Path(folder_path)
    if not path.is_dir():
        logger.warning("Checkpoint path does not exist or is not a directory: %s", path)
        return
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token or not str(hf_token).strip():
        logger.warning("HF_TOKEN not set. Skipping push to Hub.")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(path),
            repo_id=hub_model_id,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            token=hf_token or True,
        )
        logger.info("Pushed checkpoint to %s/%s", hub_model_id, path_in_repo)
    except Exception as e:
        logger.warning("Failed to push checkpoint to Hub: %s", e)
