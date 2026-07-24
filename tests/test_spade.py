# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for SPADE Imaginaire integration wrapper."""

from pathlib import Path


def _make_fake_imaginaire_root(root: Path) -> Path:
    imaginaire_root = root / "imaginaire_src"
    imaginaire_root.mkdir()
    (imaginaire_root / "imaginaire").mkdir()
    (imaginaire_root / "imaginaire" / "__init__.py").write_text("", encoding="utf-8")
    (imaginaire_root / "inference.py").write_text("print('ok')\n", encoding="utf-8")
    return imaginaire_root


def test_ensure_imaginaire_path_explicit(tmp_path):
    from src.pipelines.spade import _ensure_imaginaire_path

    src_root = _make_fake_imaginaire_root(tmp_path)
    resolved = _ensure_imaginaire_path(src_root)
    assert resolved == src_root.resolve()


def test_spade_pipeline_builds_command(tmp_path):
    from src.pipelines.spade import SPADEPipeline

    src_root = _make_fake_imaginaire_root(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("dummy: true\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"ckpt")

    pipe = SPADEPipeline(
        config_path=config,
        checkpoint_path=ckpt,
        imaginaire_src_path=src_root,
        python_executable="python3",
    )
    cmd = pipe._build_command(
        tmp_path / "out",
        seed=7,
        single_gpu=True,
        local_rank=2,
        num_workers=4,
        logdir=tmp_path / "logs",
        debug=True,
        extra_args=["--foo", "bar"],
    )
    assert cmd[0] == "python3"
    assert "--config" in cmd
    assert str(config.resolve()) in cmd
    assert "--checkpoint" in cmd
    assert str(ckpt.resolve()) in cmd
    assert "--single_gpu" in cmd
    assert "--debug" in cmd
    assert cmd[-2:] == ["--foo", "bar"]


def test_spade_pipeline_runs_subprocess(monkeypatch, tmp_path):
    from src.pipelines.spade import SPADEPipeline, SPADEPipelineOutput

    src_root = _make_fake_imaginaire_root(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("dummy: true\n", encoding="utf-8")

    captured = {}

    class _Result:
        stdout = "out"
        stderr = "err"

    def _fake_run(command, cwd, env, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return _Result()

    monkeypatch.setattr("src.pipelines.spade.subprocess.run", _fake_run)

    pipe = SPADEPipeline(config_path=config, imaginaire_src_path=src_root)
    out = pipe(
        output_dir=tmp_path / "outputs",
        capture_output=True,
        extra_args=["--bar"],
        env={"TEST_ENV": "1"},
    )
    assert isinstance(out, SPADEPipelineOutput)
    assert out.output_dir.exists()
    assert captured["cwd"] == str(src_root.resolve())
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is True
    assert "--bar" in captured["command"]
