# Credits: Hub integration utilities for training checkpoints.

"""Training utilities: checkpoint upload to Hugging Face Hub and Storage Buckets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_bucket_exists(bucket_id: str, private: bool = False, token: str | None = None) -> None:
    """Ensure bucket exists. Creates under user or org based on bucket_id."""
    try:
        from huggingface_hub import create_bucket

        # "username/bucket" or "org/bucket" -> full id; "bucket" -> under current user
        create_bucket(bucket_id, private=private, exist_ok=True, token=token)
    except Exception as e:
        logger.debug("Bucket create (may already exist): %s", e)


def push_checkpoint_to_bucket(
    folder_path: str | Path,
    bucket_id: str,
    path_in_bucket: str,
    token: str | None = None,
) -> None:
    """Sync a checkpoint directory to Hugging Face Storage Bucket.

    Uses sync_bucket for efficient incremental uploads with Xet deduplication.
    Requires huggingface_hub >= 1.5.0.

    Parameters
    ----------
    folder_path : str | Path
        Local path to the checkpoint directory (e.g. ``checkpoint-epoch-50``).
    bucket_id : str
        Bucket id (e.g. ``"username/my-training-bucket"``).
    path_in_bucket : str
        Destination path within the bucket (e.g. ``"cut/facades/checkpoint-50"``).
    token : str | None
        HF token. If None, uses ``HF_TOKEN`` env var or cached token.
    """
    path = Path(folder_path)
    if not path.is_dir():
        logger.warning("Checkpoint path does not exist or is not a directory: %s", path)
        return
    token = token or os.environ.get("HF_TOKEN")
    if not token or not str(token).strip():
        logger.warning("HF_TOKEN not set. Skipping push to Storage Bucket.")
        return
    try:
        from huggingface_hub import create_bucket, sync_bucket

        _ensure_bucket_exists(bucket_id, token=token)
        handle = f"hf://buckets/{bucket_id}"
        dest = f"{handle.rstrip('/')}/{path_in_bucket}".rstrip("/") if path_in_bucket else handle
        sync_bucket(str(path), dest, token=token)
        logger.info("Synced checkpoint to %s/%s", bucket_id, path_in_bucket)
    except ImportError:
        logger.warning(
            "huggingface_hub >= 1.5.0 required for Storage Buckets. "
            "Upgrade with: pip install -U huggingface_hub"
        )
    except Exception as e:
        logger.warning("Failed to push checkpoint to Storage Bucket: %s", e)


def sync_tensorboard_to_bucket(
    tensorboard_dir: str | Path,
    bucket_id: str,
    path_in_bucket: str,
    token: str | None = None,
) -> None:
    """Sync TensorBoard event files to Hugging Face Storage Bucket.

    Parameters
    ----------
    tensorboard_dir : str | Path
        Local path to the tensorboard log directory.
    bucket_id : str
        Bucket id (e.g. ``"username/my-training-bucket"``).
    path_in_bucket : str
        Destination path within the bucket (e.g. ``"cut/facades/tensorboard"``).
    token : str | None
        HF token. If None, uses ``HF_TOKEN`` env var or cached token.
    """
    path = Path(tensorboard_dir)
    if not path.is_dir():
        logger.warning("TensorBoard dir does not exist: %s", path)
        return
    token = token or os.environ.get("HF_TOKEN")
    if not token or not str(token).strip():
        logger.warning("HF_TOKEN not set. Skipping sync to Storage Bucket.")
        return
    try:
        from huggingface_hub import sync_bucket

        _ensure_bucket_exists(bucket_id, token=token)
        handle = f"hf://buckets/{bucket_id}"
        dest = f"{handle.rstrip('/')}/{path_in_bucket}".rstrip("/") if path_in_bucket else handle
        sync_bucket(str(path), dest, token=token)
        logger.info("Synced TensorBoard logs to %s/%s", bucket_id, path_in_bucket)
    except ImportError:
        logger.warning(
            "huggingface_hub >= 1.5.0 required for Storage Buckets. "
            "Upgrade with: pip install -U huggingface_hub"
        )
    except Exception as e:
        logger.warning("Failed to sync TensorBoard to Storage Bucket: %s", e)


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
