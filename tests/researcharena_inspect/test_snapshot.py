"""The final workspace kept in the log: archiving, the size limit, restoring, staging (no Docker).

The Docker tests that compare the restored folder with upstream's pipeline are in
`test_scoring_parity.py` and `test_snapshot_docker.py`.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from inspect_ai.log import EvalSample
from inspect_ai.util import ExecResult, Store

from researcharena_inspect.agents import write_run_dir
from researcharena_inspect.monitor import local_bundle, staged_files
from researcharena_inspect.monitor.staging import workspace_files
from researcharena_inspect.scoring import snapshot_archive
from researcharena_inspect.scoring.snapshot import (
    SNAPSHOT_NAME,
    TaskSnapshot,
    keep_task_snapshot,
    restore_snapshot,
    snapshot_bytes,
)
from tests.researcharena_inspect.fake_sandbox import Call, FakeSandbox
from tests.researcharena_inspect.test_agents_record import claude_sample
from tests.researcharena_inspect.tree_manifest import describe


def _make_workspace(root: Path) -> None:
    """The kinds of entry a workspace can hold."""
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "rmsnorm.cu").write_text("// tuned\n")
    (root / "run.sh").write_text("#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    (root / "secret.txt").write_text("agent only\n")
    (root / "secret.txt").chmod(0o600)
    (root / "objects" / "ab").mkdir(parents=True)
    (root / "objects" / "ab" / "cdef").write_bytes(bytes(range(256)) * 40)
    (root / "objects" / "ab" / "cdef").chmod(0o444)
    (root / "locked").mkdir()
    (root / "locked" / "inside.txt").write_text("x")
    (root / "locked").chmod(0o555)
    (root / "empty").mkdir(mode=0o700)
    (root / "kernel_link").symlink_to("kernels/rmsnorm.cu")
    (root / "model").symlink_to("/home/user/hf_cache/models--x")
    (root / "kernels_dir").symlink_to("kernels", target_is_directory=True)
    (root / "dangling").symlink_to("nowhere")
    os.mkfifo(root / "pipe")


def _unlock(root: Path) -> None:
    for path in (root / "locked", root / SNAPSHOT_NAME / "locked"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)


def _entries(root: Path) -> dict:
    return describe(str(root))["entries"]


def test_archive_and_restore_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_workspace(source)
    out = tmp_path / "snapshot.tar.gz"
    try:
        report = snapshot_archive.archive(str(source), SNAPSHOT_NAME, str(out), 10_000_000)
        assert report["status"] == "kept"
        assert report["archive_bytes"] == out.stat().st_size
        assert report["skipped"] == [["pipe", "fifo"]]
        assert report["links"] == 4
        assert report["files"] == 5
        assert report["largest"][0] == ["objects/ab/cdef", 10240]

        restored = tmp_path / "red_team"
        result = restore_snapshot(out.read_bytes(), restored)
        assert result["skipped"] == []
        assert result["dropped_bits"] == []
        (source / "pipe").unlink()  # not archived; describing it would block on the FIFO
        expected = _entries(source)
        assert _entries(restored / SNAPSHOT_NAME) == expected
        # Links come back as links, their targets as written.
        assert os.readlink(restored / SNAPSHOT_NAME / "model") == "/home/user/hf_cache/models--x"
    finally:
        _unlock(source)
        _unlock(tmp_path / "red_team")


def test_archive_over_the_limit_is_not_kept(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "final_model.bin").write_bytes(os.urandom(200_000))
    out = tmp_path / "snapshot.tar.gz"
    report = snapshot_archive.archive(str(source), SNAPSHOT_NAME, str(out), 50_000)
    assert report["status"] == "too_large"
    assert report["archive_bytes"] is None
    assert report["raw_bytes"] == 200_000
    assert report["largest"] == [["final_model.bin", 200_000]]
    assert not out.exists()


def test_archiver_refuses_a_missing_folder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = snapshot_archive.main(["x", str(tmp_path / "nope"), SNAPSHOT_NAME, "out", "10"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def _tar(*members: tuple[tarfile.TarInfo, bytes | None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


def _dir(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _file(name: str, data: bytes = b"x", mode: int = 0o644) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.mode = mode
    return info, data


def _link(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


@pytest.mark.parametrize(
    "members",
    [
        [_dir(SNAPSHOT_NAME), _file(f"{SNAPSHOT_NAME}/../escape")],
        [_dir(SNAPSHOT_NAME), _file("/etc/escape")],
        [_dir(SNAPSHOT_NAME), _file("elsewhere/x")],
        [_file(SNAPSHOT_NAME)],
        [_dir(SNAPSHOT_NAME), _link(f"{SNAPSHOT_NAME}/up", "/tmp"), _file(f"{SNAPSHOT_NAME}/up/x")],
        [_dir(SNAPSHOT_NAME), _file(f"{SNAPSHOT_NAME}/a"), _file(f"{SNAPSHOT_NAME}/a")],
        [_dir(SNAPSHOT_NAME), _file(f"{SNAPSHOT_NAME}/sub/x")],
    ],
    ids=["dotdot", "absolute", "other-top", "top-not-folder", "through-link", "twice", "no-folder"],
)
def test_restore_refuses_unsafe_archives(tmp_path: Path, members: list) -> None:
    with pytest.raises(ValueError, match=r"archive|folder"):
        restore_snapshot(_tar(*members), tmp_path / "red_team")
    assert not (tmp_path / "escape").exists()


def test_restore_drops_setuid_bits(tmp_path: Path) -> None:
    archive = _tar(_dir(SNAPSHOT_NAME), _file(f"{SNAPSHOT_NAME}/tool", mode=0o4755))
    report = restore_snapshot(archive, tmp_path)
    assert report["dropped_bits"] == [f"{SNAPSHOT_NAME}/tool"]
    assert (tmp_path / SNAPSHOT_NAME / "tool").stat().st_mode & 0o7777 == 0o755


def test_restore_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / SNAPSHOT_NAME).mkdir()
    with pytest.raises(FileExistsError):
        restore_snapshot(_tar(_dir(SNAPSHOT_NAME)), tmp_path)


def _kept(sample: EvalSample, archive: bytes) -> EvalSample:
    store = Store(dict(sample.store))
    record = TaskSnapshot(store=store)
    record.status = "kept"
    record.archive = base64.b64encode(archive).decode()
    record.archive_bytes = len(archive)
    record.archive_sha256 = hashlib.sha256(archive).hexdigest()
    return sample.model_copy(update={"store": dict(store.items())})


def test_write_run_dir_restores_the_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_workspace(source)
    out = tmp_path / "snapshot.tar.gz"
    try:
        snapshot_archive.archive(str(source), SNAPSHOT_NAME, str(out), 10_000_000)
        sample = _kept(claude_sample(), out.read_bytes())
        run_dir = write_run_dir(sample, tmp_path / "runs")
        snapshot = run_dir / "red_team" / SNAPSHOT_NAME
        (source / "pipe").unlink()
        expected = _entries(source)
        assert _entries(snapshot) == expected
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["inspect_task_snapshot"]["status"] == "kept"
        assert "archive" not in meta["inspect_task_snapshot"]
        assert local_bundle(run_dir).workspace == snapshot
    finally:
        _unlock(source)
        for red in (tmp_path / "runs").glob("*/red_team"):
            _unlock(red)


def test_write_run_dir_without_a_kept_snapshot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sample = claude_sample()
    store = Store(dict(sample.store))
    record = TaskSnapshot(store=store)
    record.status = "too_large"
    record.raw_bytes = 3_000_000_000
    sample = sample.model_copy(update={"store": dict(store.items())})
    run_dir = write_run_dir(sample, tmp_path)
    assert not (run_dir / "red_team" / SNAPSHOT_NAME).exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["inspect_task_snapshot"]["status"] == "too_large"
    assert meta["inspect_task_snapshot"]["raw_bytes"] == 3_000_000_000
    assert "too_large" in caplog.text
    assert local_bundle(run_dir).workspace is None


def test_damaged_archive_is_refused(tmp_path: Path) -> None:
    archive = _tar(_dir(SNAPSHOT_NAME))
    sample = _kept(claude_sample(), archive)
    record = sample.store_as(TaskSnapshot)
    record.archive_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        snapshot_bytes(record)


def test_local_workspace_links_leaving_it_are_not_followed(tmp_path: Path) -> None:
    host_secret = tmp_path / "host_secret.txt"
    host_secret.write_text("not for the monitor\n")
    workspace = tmp_path / "red_team" / SNAPSHOT_NAME
    (workspace / "kernels").mkdir(parents=True)
    (workspace / "kernels" / "a.cu").write_text("kernel\n")
    (workspace / "inside").symlink_to("kernels/a.cu")
    (workspace / "absolute").symlink_to(host_secret)
    (workspace / "relative").symlink_to("../../host_secret.txt")

    local = [rel for rel, _ in workspace_files(workspace)]
    assert local == ["inside", "kernels/a.cu"]
    published = [rel for rel, _ in workspace_files(workspace, links_outside=True)]
    assert published == ["absolute", "inside", "relative", "kernels/a.cu"]

    transcript = tmp_path / "t.txt"
    transcript.write_text("t\n")
    staged = staged_files(
        transcript=transcript,
        main_task_instruction="i",
        prompt="p",
        workspace=workspace,
        with_artifact=True,
    )
    assert "task/absolute" not in {s.path for s in staged}


def _box(monkeypatch: pytest.MonkeyPatch, fake: FakeSandbox) -> None:
    monkeypatch.setattr("researcharena_inspect.scoring.snapshot.evaluator_sandbox", lambda: fake)
    monkeypatch.setattr("researcharena_inspect.environment.staging.sandbox_env", lambda: fake)


def test_keep_task_snapshot_records_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(call: Call) -> ExecResult[str] | None:
        if call.cmd[0] == "python3":
            return ExecResult(False, 1, "", "python3: not found")
        return None

    _box(monkeypatch, FakeSandbox(respond=respond))
    store = Store()
    record = asyncio.run(keep_task_snapshot(store, 1000))
    assert record.status == "error"
    assert "python3: not found" in (record.error or "")
    assert store.get("TaskSnapshot:status") == "error"


def test_keep_task_snapshot_keeps_the_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _tar(_dir(SNAPSHOT_NAME))
    fake = FakeSandbox()

    def respond(call: Call) -> ExecResult[str] | None:
        if call.cmd[0] == "python3":
            assert call.user == "root"
            assert call.cmd[2:4] == ["/mnt/eval/red_team/task_snapshot", SNAPSHOT_NAME]
            fake.files[call.cmd[4]] = archive
            report = {"status": "kept", "archive_bytes": len(archive), "files": 0, "dirs": 1}
            return ExecResult(True, 0, json.dumps(report) + "\n", "")
        return None

    fake.respond = respond
    _box(monkeypatch, fake)
    store = Store()
    record = asyncio.run(keep_task_snapshot(store, 1000))
    assert record.status == "kept"
    assert snapshot_bytes(record) == archive
    assert record.summary()["dirs"] == 1
    assert "archive" not in record.summary()
    # Processes left by the evaluator are stopped first; the scratch folder is removed last.
    scripts = [c.script for c in fake.calls]
    assert "kill -KILL" in scripts[0]
    assert fake.calls[-1].cmd == ["rm", "-rf", "/var/tmp/researcharena-snapshot"]


def test_keep_task_snapshot_off() -> None:
    store = Store()
    record = asyncio.run(keep_task_snapshot(store, 1000, enabled=False))
    assert record.status == "off"
    assert store.get("TaskSnapshot:status") == "off"
