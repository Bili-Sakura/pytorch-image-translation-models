# Copyright (c) 2026 EarthBridge Team.
# Credits: SPADE/GauGAN (Park et al., CVPR 2019) via NVLabs Imaginaire - https://github.com/NVlabs/imaginaire
#
"""SPADE integration pipeline backed by NVLabs Imaginaire inference."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput


def _ensure_imaginaire_path(imaginaire_src_path: Optional[str | Path]) -> Path:
    """Resolve Imaginaire source path containing ``inference.py``."""
    if imaginaire_src_path is not None:
        path = Path(imaginaire_src_path).expanduser()
        if (path / "inference.py").exists() and (path / "imaginaire").exists():
            return path.resolve()
        raise FileNotFoundError(
            f"Imaginaire source not found at {path}. Expected inference.py and imaginaire/ directory."
        )

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "imaginaire",
        root / "Imaginaire",
        root / "_upstream_imaginaire",
        root / "projects" / "imaginaire",
        Path.cwd() / "imaginaire",
        Path.cwd() / "Imaginaire",
        Path.cwd() / "_upstream_imaginaire",
        Path.cwd() / "projects" / "imaginaire",
    ]
    for candidate in candidates:
        if (candidate / "inference.py").exists() and (candidate / "imaginaire").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Imaginaire source not found. Clone https://github.com/NVlabs/imaginaire.git, "
        "set imaginaire_src_path, or place it under ./imaginaire or ./_upstream_imaginaire."
    )


@dataclass
class SPADEPipelineOutput(BaseOutput):
    """Output of SPADE pipeline invocation."""

    output_dir: Path
    command: List[str]
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class SPADEPipeline(DiffusionPipeline):
    """Wrapper around Imaginaire SPADE inference entrypoint."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        imaginaire_src_path: Optional[str | Path] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"SPADE config not found: {self.config_path}")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SPADE checkpoint not found: {self.checkpoint_path}")

        self.imaginaire_src_path = _ensure_imaginaire_path(imaginaire_src_path)
        self.python_executable = python_executable or sys.executable

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config_filename: str = "config.yaml",
        checkpoint_filename: Optional[str] = None,
        imaginaire_src_path: Optional[str | Path] = None,
        python_executable: Optional[str] = None,
    ) -> "SPADEPipeline":
        """Load SPADE integration from a local method directory."""
        root = Path(pretrained_model_name_or_path).expanduser()
        config_path = root / config_filename
        checkpoint_path = (root / checkpoint_filename) if checkpoint_filename else None
        return cls(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            imaginaire_src_path=imaginaire_src_path,
            python_executable=python_executable,
        )

    def _build_command(
        self,
        output_dir: Path,
        *,
        seed: int = 0,
        single_gpu: bool = True,
        local_rank: int = 0,
        num_workers: Optional[int] = None,
        logdir: Optional[str | Path] = None,
        debug: bool = False,
        extra_args: Optional[Sequence[str]] = None,
    ) -> List[str]:
        command: List[str] = [
            self.python_executable,
            str(self.imaginaire_src_path / "inference.py"),
            "--config",
            str(self.config_path),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--local_rank",
            str(local_rank),
        ]
        if self.checkpoint_path is not None:
            command.extend(["--checkpoint", str(self.checkpoint_path)])
        if logdir is not None:
            command.extend(["--logdir", str(Path(logdir).expanduser())])
        if num_workers is not None:
            command.extend(["--num_workers", str(num_workers)])
        if single_gpu:
            command.append("--single_gpu")
        if debug:
            command.append("--debug")
        if extra_args:
            command.extend(list(extra_args))
        return command

    def __call__(
        self,
        *,
        output_dir: str | Path,
        seed: int = 0,
        single_gpu: bool = True,
        local_rank: int = 0,
        num_workers: Optional[int] = None,
        logdir: Optional[str | Path] = None,
        debug: bool = False,
        capture_output: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        return_dict: bool = True,
    ) -> Union[SPADEPipelineOutput, tuple]:
        """Run Imaginaire SPADE inference and return output metadata."""
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        command = self._build_command(
            output_path,
            seed=seed,
            single_gpu=single_gpu,
            local_rank=local_rank,
            num_workers=num_workers,
            logdir=logdir,
            debug=debug,
            extra_args=extra_args,
        )

        run_env = os.environ.copy()
        if env is not None:
            run_env.update(env)

        result = subprocess.run(
            command,
            cwd=str(self.imaginaire_src_path),
            env=run_env,
            capture_output=capture_output,
            text=capture_output,
            check=True,
        )

        output = SPADEPipelineOutput(
            output_dir=output_path,
            command=command,
            stdout=result.stdout if capture_output else None,
            stderr=result.stderr if capture_output else None,
        )
        if not return_dict:
            return (output.output_dir, output.command, output.stdout, output.stderr)
        return output


def load_spade_pipeline(
    config_path: str | Path,
    *,
    checkpoint_path: Optional[str | Path] = None,
    imaginaire_src_path: Optional[str | Path] = None,
    python_executable: Optional[str] = None,
) -> SPADEPipeline:
    """Create a SPADE pipeline wrapper around Imaginaire inference."""
    return SPADEPipeline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        imaginaire_src_path=imaginaire_src_path,
        python_executable=python_executable,
    )
