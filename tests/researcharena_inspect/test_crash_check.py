"""Part D's diagnostic task and its in-sandbox script, without a GPU.

`kernel_bench.py` runs in the sandbox with PyTorch; here only its kernel selection, fault
attribution and result line are checked, with a stand-in for the compiled module.
"""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inspect_ai.solver import Generate
from inspect_ai.util import SandboxEnvironmentSpec

from researcharena_inspect.environment.sandbox import COMPOSE_FILE
from researcharena_inspect.scoring.crash_check import (
    PARTS,
    bench_result,
    bench_runs,
    kernel_crash_check,
    summarize,
)
from researcharena_inspect.scoring.kernel_bench import (
    KERNELS,
    RESULT_PREFIX,
    OnlyKernels,
    parse_kernels,
)


def _ops() -> Any:
    def fault(*args: Any) -> Any:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    return SimpleNamespace(
        rmsnorm=lambda x, w, e: "rmsnorm out",
        rotary_embedding=lambda q, p, b: "rope out",
        fused_mlp=lambda x, g, u: "mlp out",
        attention=fault,
    )


def test_kernel_selection() -> None:
    assert parse_kernels("all") == list(KERNELS)
    assert parse_kernels("none") == []
    assert parse_kernels("attention,rmsnorm") == ["rmsnorm", "attention"]
    with pytest.raises(ValueError, match="unknown kernels"):
        parse_kernels("rmsnorm,flash")


def test_only_the_chosen_kernels_are_visible() -> None:
    # inject_agent_kernels asks hasattr(agent_ops, name) for each kernel (inject.py:17-18).
    visible = OnlyKernels(_ops(), ["rmsnorm"])
    assert hasattr(visible, "rmsnorm")
    for name in ("rotary_embedding", "fused_mlp", "attention"):
        assert not hasattr(visible, name)
    assert dir(visible) == ["rmsnorm"]
    assert not OnlyKernels(_ops(), []).__dir__()


def test_sync_each_names_the_faulting_call_and_records_shapes() -> None:
    synced: list[int] = []
    visible = OnlyKernels(
        _ops(), ["rmsnorm", "attention"], sync_each=True, synchronize=lambda: synced.append(1)
    )
    tensor = SimpleNamespace(shape=(1, 32, 7, 64), dtype="torch.float16")
    assert visible.rmsnorm(tensor, tensor, 1e-5) == "rmsnorm out"
    assert synced == [1]
    kv = SimpleNamespace(shape=(1, 4, 7, 64), dtype="torch.float16")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        visible.attention(tensor, kv, kv, True)
    assert visible.calls == {"rmsnorm": 1, "attention": 1}
    assert visible.fault == {
        "kernel": "attention",
        "call": 1,
        "args": [
            {"shape": [1, 32, 7, 64], "dtype": "torch.float16"},
            {"shape": [1, 4, 7, 64], "dtype": "torch.float16"},
            {"shape": [1, 4, 7, 64], "dtype": "torch.float16"},
            True,
        ],
        "error": "RuntimeError: CUDA error: an illegal memory access was encountered",
    }
    assert visible.fault is not None
    assert visible.shapes["attention"] == [visible.fault["args"]]


def test_runs_cover_each_kernel_alone_twice_and_under_launch_blocking() -> None:
    runs = bench_runs()
    plain = [r["kernels"] for r in runs if not r.get("flags")]
    assert plain == 2 * ["none", *KERNELS, "all"]
    blocking = [r for r in runs if r.get("env") == {"CUDA_LAUNCH_BLOCKING": "1"}]
    assert [r["kernels"] for r in blocking] == [*KERNELS, "all"]
    assert all(r["flags"] == ["--sync-each"] for r in blocking)
    assert [r["kernels"] for r in runs if r.get("flags") == ["--vanilla-first"]] == ["all"]


def test_result_line_is_the_last_one() -> None:
    out = "noise\n" + RESULT_PREFIX + '{"a": 1}\nmore\n' + RESULT_PREFIX + '{"a": 2}\n'
    assert bench_result(out) == {"a": 2}
    assert bench_result("no result") is None
    assert bench_result(RESULT_PREFIX + "{not json") is None


def test_summary_of_each_part() -> None:
    repeats = {
        "part": "plain-repeats",
        "skipped": [],
        "repeats": [
            {
                "metrics_json": '{"main_task_score": 0.0}',
                "attempts": [{"log": "illegal memory access"}],
            },
            {"metrics_json": '{"main_task_score": 1.2}', "attempts": [{"log": "SPEEDUP"}]},
        ],
    }
    assert summarize(repeats)["illegal_memory_access"] == [True, False]
    isolation = {
        "part": "kernel-isolation",
        "skipped": ["compute-sanitizer memcheck: attention"],
        "build": {"compile_status": "PASS"},
        "bench_runs": [
            {
                "label": "none (run 1)",
                "result": {"status": "ok", "benchmark": {"avg_tokens_per_sec": 60.0}},
            },
            {"label": "attention (run 1)", "result": {"status": "error", "error": "x"}},
        ],
        "sanitizer_runs": [],
    }
    summary = summarize(isolation)
    assert summary["build"] == "PASS"
    assert [(r["label"], r["status"], r["tokens_per_sec"]) for r in summary["runs"]] == [
        ("none (run 1)", "ok", 60.0),
        ("attention (run 1)", "error", None),
    ]
    json.dumps(summary)


def test_task_has_one_sample_per_part() -> None:
    task = kernel_crash_check()
    assert [s.id for s in task.dataset] == list(PARTS)
    assert task.time_limit == 3600
    # Docker by default, one container.
    assert task.sandbox == SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
    assert [s.id for s in kernel_crash_check(parts="kernel-isolation").dataset] == [
        "kernel-isolation"
    ]
    with pytest.raises(ValueError, match="unknown parts"):
        kernel_crash_check(parts=["everything"])


def _fake_modules(monkeypatch: pytest.MonkeyPatch, *, fault_in: str | None) -> dict[str, Any]:
    """Stand-ins for torch, transformers and upstream's bench, config and inject modules."""
    import sys
    import types

    seen: dict[str, Any] = {"loads": [], "synced": 0, "models": 0}
    torch = types.ModuleType("torch")
    torch.__version__ = "2.fake"  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(cuda="12.fake")  # type: ignore[attr-defined]

    def synchronize() -> None:
        seen["synced"] += 1

    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: False, synchronize=synchronize, empty_cache=lambda: None
    )

    def load(**kwargs: Any) -> Any:
        seen["loads"].append(kwargs)

        def kernel(name: str) -> Any:
            def run(*args: Any) -> str:
                if name == fault_in:
                    raise RuntimeError("CUDA error: an illegal memory access was encountered")
                return name

            return run

        return SimpleNamespace(**{k: kernel(k) for k in KERNELS})

    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.load = load  # type: ignore[attr-defined]
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.fake"  # type: ignore[attr-defined]

    bench = types.ModuleType("bench")

    def load_model_and_tokenizer(config: Any) -> Any:
        seen["models"] += 1
        return SimpleNamespace(), SimpleNamespace()

    def benchmark_generation(model: Any, tokenizer: Any, config: Any, num_prompts: Any) -> Any:
        for name in KERNELS:
            if hasattr(model, name):
                getattr(model, name)(SimpleNamespace(shape=(1, 7, 64), dtype="f16"))
        return {"avg_tokens_per_sec": 50.0, "num_prompts": num_prompts}

    bench.apply_seed = lambda seed: None  # type: ignore[attr-defined]
    bench.load_model_and_tokenizer = load_model_and_tokenizer  # type: ignore[attr-defined]
    bench.benchmark_generation = benchmark_generation  # type: ignore[attr-defined]
    config = types.ModuleType("config")
    config.ArenaConfig = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        seed=42, max_new_tokens=128, warmup_runs=3, eval_prompts=["a"] * 8
    )
    inject = types.ModuleType("inject")

    def inject_agent_kernels(model: Any, ops: Any, verbose: bool = False) -> dict[str, bool]:
        injected = {}
        for name in KERNELS:
            injected[name] = hasattr(ops, name)
            if injected[name]:
                setattr(model, name, getattr(ops, name))
        return injected

    inject.inject_agent_kernels = inject_agent_kernels  # type: ignore[attr-defined]
    for name, module in {
        "torch": torch,
        "torch.utils": types.ModuleType("torch.utils"),
        "torch.utils.cpp_extension": cpp_extension,
        "transformers": transformers,
        "bench": bench,
        "config": config,
        "inject": inject,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return seen


def _main(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    from researcharena_inspect.scoring import kernel_bench

    code = kernel_bench.main(args)
    out = capsys.readouterr().out
    result = bench_result(out)
    assert result is not None, out
    return code, result


def test_bench_injects_only_the_chosen_kernels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    seen = _fake_modules(monkeypatch, fault_in="attention")
    common = ["--task", str(tmp_path), "--build-dir", str(tmp_path / "build")]
    code, result = _main(["bench", "--kernels", "rmsnorm", "--vanilla-first", *common], capsys)
    assert code == 0
    assert result["status"] == "ok"
    assert result["injected"] == {
        "rmsnorm": True,
        "rotary_embedding": False,
        "fused_mlp": False,
        "attention": False,
    }
    assert result["benchmark"]["avg_tokens_per_sec"] == 50.0
    assert result["vanilla"]["avg_tokens_per_sec"] == 50.0
    assert seen["models"] == 2  # vanilla first, then the injected model
    assert (result["torch"], result["transformers"]) == ("2.fake", "4.fake")
    [build] = seen["loads"]
    assert build["name"] == "agent_ops"
    assert build["extra_cuda_cflags"] == ["-O3", "--use_fast_math", "-lineinfo"]
    assert build["build_directory"] == str(tmp_path / "build")


def test_bench_attributes_a_fault_with_sync_each(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    seen = _fake_modules(monkeypatch, fault_in="attention")
    common = ["--task", str(tmp_path), "--build-dir", str(tmp_path / "build")]
    args = ["bench", "--kernels", "all", "--sync-each", "--max-new-tokens", "4", *common]
    code, result = _main(args, capsys)
    assert code == 1
    assert result["status"] == "error"
    assert "illegal memory access" in result["error"]
    assert result["fault"]["kernel"] == "attention"
    assert result["max_new_tokens"] == 4
    assert result["calls"] == {"rmsnorm": 1, "rotary_embedding": 1, "fused_mlp": 1, "attention": 1}
    assert seen["synced"] == 3  # after each kernel that returned


def test_build_reports_a_failed_compile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    _fake_modules(monkeypatch, fault_in=None)
    import sys

    def broken(**kwargs: Any) -> Any:
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr(sys.modules["torch.utils.cpp_extension"], "load", broken)
    code, result = _main(["build", "--task", str(tmp_path), "--build-dir", str(tmp_path)], capsys)
    assert code == 1
    assert result["compile_status"] == "FAIL"
    assert result["error"] == "nvcc failed"


def _crash_check_state(part: str) -> Any:
    from inspect_ai.model import ModelName
    from inspect_ai.solver import TaskState

    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=part,
        epoch=1,
        input="x",
        messages=[],
        metadata={"part": part},
    )


async def _generate_nothing(state: Any, **kwargs: Any) -> Any:
    return state


_no_generate = cast(Generate, _generate_nothing)


def _patch_crash_check(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    from researcharena_inspect.environment.evaluation import EvaluationCommand
    from researcharena_inspect.scoring import crash_check
    from researcharena_inspect.scoring.scorer import Attempt, EvaluationRun
    from tests.researcharena_inspect.fake_sandbox import FakeSandbox

    order: list[str] = []
    fake = FakeSandbox()
    original = fake.exec

    async def exec_(cmd: list[str], **kwargs: Any) -> Any:
        if "rm -rf /mnt/eval/red_team" in " ".join(cmd):
            order.append("clear")
        return await original(cmd, **kwargs)

    fake.exec = exec_  # type: ignore[method-assign]

    async def stage(*args: Any, **kwargs: Any) -> Any:
        order.append("stage")
        return SimpleNamespace(hook_outputs=[])

    async def prepare(setting: str, side_task: str) -> Any:
        order.append(f"prepare {setting} {side_task}")
        return EvaluationCommand(["python3"], "/home/evaluator", {}, "user", 3600, 3, "/m")

    async def evaluate(command: Any) -> Any:
        order.append("evaluate")
        attempt = Attempt(1, 0, 250.0, True, "illegal memory access", 21)
        return EvaluationRun(["python3"], "/", "user", 0.0, [attempt], '{"main": 0.0}', False)

    monkeypatch.setattr(crash_check, "sandbox", lambda: fake)
    monkeypatch.setattr(crash_check, "stage_workspace", stage)
    monkeypatch.setattr(crash_check, "prepare_evaluation", prepare)
    monkeypatch.setattr(crash_check, "run_evaluation", evaluate)
    return fake, order


async def test_run_crash_check_repeats_the_plain_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researcharena_inspect.scoring.crash_check import METADATA_KEY, run_crash_check

    _, order = _patch_crash_check(monkeypatch)
    state = await run_crash_check(repeats=2)(_crash_check_state("plain-repeats"), _no_generate)
    # The evaluation folders are cleared before each preparation.
    assert order == [
        "stage",
        "clear",
        "prepare cuda_kernel none",
        "evaluate",
        "clear",
        "prepare cuda_kernel none",
        "evaluate",
    ]
    result = state.metadata[METADATA_KEY]
    assert [r["repeat"] for r in result["repeats"]] == [1, 2]
    assert result["summary"]["illegal_memory_access"] == [True, True]
    assert result["skipped"] == []


async def test_run_crash_check_skips_what_would_overrun(monkeypatch: pytest.MonkeyPatch) -> None:
    from researcharena_inspect.scoring.crash_check import METADATA_KEY, run_crash_check

    _, order = _patch_crash_check(monkeypatch)
    # 1000 s leaves less than three attempts' worth for a repeat, so none starts.
    state = await run_crash_check(repeats=3, time_limit=1000)(
        _crash_check_state("plain-repeats"), _no_generate
    )
    assert "evaluate" not in order
    result = state.metadata[METADATA_KEY]
    assert result["skipped"] == [f"plain evaluation {n}" for n in (1, 2, 3)]


async def test_run_crash_check_stops_after_a_failed_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researcharena_inspect.scoring.crash_check import METADATA_KEY, run_crash_check

    fake, _ = _patch_crash_check(monkeypatch)
    failed_build = RESULT_PREFIX + json.dumps({"compile_status": "FAIL", "error": "nvcc"})
    fake.respond = lambda call: (
        SimpleNamespace(returncode=1, stdout=failed_build, stderr="", success=False)
        if "build" in call.cmd
        else None
    )
    state = await run_crash_check()(_crash_check_state("kernel-isolation"), _no_generate)
    result = state.metadata[METADATA_KEY]
    assert result["build"] == {"compile_status": "FAIL", "error": "nvcc"}
    assert result["bench_runs"] == []
    steps = [s["step"] for s in result["steps"]]
    assert steps[-1] == "build"
    [build] = [s for s in result["steps"] if s["step"] == "build"]
    assert build["user"] == "user"
    assert build["cwd"] == "/tmp/kernel_check"
    assert build["cmd"][:3] == ["python3", "/tmp/kernel_check/kernel_bench.py", "build"]
    assert "/tmp/kernel_check/kernel_bench.py" in fake.files
