import importlib

import pytest
from inspect_ai import eval
from inspect_ai.util import ExecResult
from pytest_inspect_evals.assertions import assert_eval_success

from researcharena_inspect import gpu_probe
from researcharena_inspect.gpu_probe import K8S_VALUES, NVIDIA_SMI, run_nvidia_smi

# The package exports the task under the module's own name, so fetch the module itself.
probe_module = importlib.import_module("researcharena_inspect.gpu_probe")


class FakeSandbox:
    def __init__(self, result: ExecResult[str] | Exception) -> None:
        self.result = result
        self.commands: list[list[str]] = []

    async def exec(self, cmd: list[str], **kwargs: object) -> ExecResult[str]:
        self.commands.append(cmd)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run_probe(
    monkeypatch: pytest.MonkeyPatch, result: ExecResult[str] | Exception
) -> tuple[str, str]:
    fake = FakeSandbox(result)
    monkeypatch.setattr(probe_module, "sandbox", lambda: fake)
    [log] = eval(
        tasks=gpu_probe(sandbox_type="local"), solver=run_nvidia_smi(), model="mockllm/model"
    )
    assert_eval_success(log, expected_sample_count=1)
    assert fake.commands == [NVIDIA_SMI]
    assert log.samples is not None
    assert log.samples[0].scores is not None
    score = log.samples[0].scores["gpu_visible"]
    return str(score.value), str(score.answer)


def test_gpu_reported_when_nvidia_smi_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    line = "NVIDIA H100 80GB HBM3, 81559 MiB, 570.86.15, 9.0\n"
    value, answer = run_probe(monkeypatch, ExecResult(True, 0, line, ""))
    assert value == "C"
    assert answer == line.strip()


def test_no_gpu_when_nvidia_smi_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    value, answer = run_probe(monkeypatch, ExecResult(False, 9, "", "NVIDIA-SMI has failed"))
    assert value == "I"
    assert answer == "NVIDIA-SMI has failed"


def test_no_gpu_when_nvidia_smi_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    value, answer = run_probe(monkeypatch, FileNotFoundError("nvidia-smi"))
    assert value == "I"
    assert "FileNotFoundError" in answer


def test_k8s_sandbox_requests_one_gpu() -> None:
    task = gpu_probe()
    assert task.sandbox is not None
    assert task.sandbox.type == "k8s"
    assert task.sandbox.config == str(K8S_VALUES)
    assert "nvidia.com/gpu: 1" in K8S_VALUES.read_text()
