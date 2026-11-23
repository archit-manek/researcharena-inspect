"""Paths in the agent's sandbox, the account the agent runs as, and its environment variables.

Follows upstream's `apptainer exec` of the agent (`src/pipeline_redteam.sh:357-378`) and of the
evaluator (`:479-499`); line numbers below are that file's. Apptainer runs as the cluster account
that started it; the port runs the container as root and the agent as `user` (D-090). `-C`
cleans the host environment, so the agent sees only the image's variables and the `--env` ones.
"""

from typing import Final

AGENT_USER: Final = "user"
"""Account the agent, its hooks and upstream's evaluator run as (D-090)."""

AGENT_UID: Final = 1000
"""uid and gid of `AGENT_USER`; upstream's is the cluster account's, unknown to us."""

AGENT_HOME: Final = "/home/user"
"""The agent's home: upstream's `$AGENT_JOB_DIR` (`--home $AGENT_JOB_DIR:/home/user`)."""

TASK_DIR: Final = f"{AGENT_HOME}/task"
"""The agent's working directory (`--pwd /home/user/task`) and the folder scored afterwards."""

PROMPT_FILE: Final = f"{AGENT_HOME}/prompt.txt"
"""The task instruction as upstream writes it (`echo "$PROMPT" > prompt.txt`, `:265`)."""

HF_HOME: Final = f"{AGENT_HOME}/hf_cache"
"""The model cache (`--env HF_HOME=/home/user/hf_cache`, `:359`)."""

AGENT_TMP: Final = "/tmp"
"""The agent's private `/tmp` (`--bind $AGENT_TMP_DIR:/tmp`, `:372`): the sandbox's own."""

PRE_AGENT_SETUP: Final = f"{AGENT_HOME}/pre_agent_setup.sh"
"""Sourced in the agent's shell right before the agent CLI starts (`:378`)."""

EVAL_DIR: Final = "/mnt/eval"
"""The run folder as upstream's evaluator sees it (`--bind $EVAL_DIR:/mnt/eval`, `:487`)."""

EVALUATOR_SERVICE: Final = "evaluator"
"""The Docker sandbox's second service, where upstream's evaluator runs (D-093): the same image
and GPU, sharing only `HF_HOME` and `EVAL_DIR` with the agent's (`sandbox.with_evaluator`)."""

EVAL_HOME: Final = "/home/user"
"""The evaluator's job folder, holding `evaluate.py`, and its home, in its own container
(`--home $EVAL_JOB_DIR:/home/user --pwd /home/user`, `:488-489`); `HF_HOME` is mounted
inside it, as upstream's overlay is."""

SHARED_EVAL_HOME: Final = "/home/evaluator"
"""The evaluator's job folder where it shares the agent's container: a sandbox with no
`evaluator` service (a custom `sandbox`, D-093)."""

CUDA_VISIBLE_DEVICES: Final = "0"

# `--env PATH=...` (`:367`): replaces the image's PATH.
AGENT_PATH: Final = (
    "/home/user/task:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)

# The image's PATH: the base image's, with arena.def's %environment prepended. Upstream's
# evaluator container (`apptainer exec -c`, no `--env PATH`) runs with it.
IMAGE_PATH: Final = (
    "/root/.local/bin:/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

# `BASH_MAX_TIMEOUT_MS` and `BASH_DEFAULT_TIMEOUT_MS` (`:368-369`).
BASH_TIMEOUT_MS: Final = "3600000"

CONTAINER_ENV: Final[dict[str, str]] = {
    "HOME": AGENT_HOME,
    "PATH": AGENT_PATH,
    "HF_HOME": HF_HOME,
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "BASH_MAX_TIMEOUT_MS": BASH_TIMEOUT_MS,
    "BASH_DEFAULT_TIMEOUT_MS": BASH_TIMEOUT_MS,
}
"""The container-wide `--env` values of `:359-369` that hold no secret, for a sandbox config
that sets them on the container (the CPU test sandbox). Every `exec` as the agent passes
`agent_env(...)`, which includes them, so the GPU sandboxes do not set them."""


def agent_env(
    *,
    redruntime: str,
    resume_agent: bool,
    agent_config: str | None = None,
    agent_effort: str | None = None,
    seed: int | None = None,
    hf_token: str = "",
) -> dict[str, str]:
    """The variables upstream sets with `--env` when it starts the agent's container.

    Pass them with every `exec` that runs as the agent: some ways of switching user reset
    `HOME`, `USER`, `LOGNAME` and `PATH`. Not included: `agents/<family>/keep_env_vars`
    (the agent runner's; no credentials, D-091) and the hooks' `AGENT_EXTRA_ENV_ARGS`
    (`staging.agent_extra_env`, D-174).

    Args:
        redruntime: the budget in hours as upstream's command line carries it
            (`prompts.format_hours`).
        resume_agent: the setting's `resume_agent` (`AGENT_RESUME`).
        agent_config: the model name (`AGENT_CONFIG`); None leaves it unset.
        agent_effort: the effort from `effort.json` (`AGENT_EFFORT`); None leaves it unset.
        seed: `TASK_SEED` and `PYTHONHASHSEED`, set only when given (`:131-138`).
        hf_token: `HF_TOKEN`.

    Returns:
        Name to value, in upstream's order.
    """
    env = {
        "HOME": AGENT_HOME,
        "USER": AGENT_USER,
        "LOGNAME": AGENT_USER,
        "HF_HOME": HF_HOME,
        "HF_TOKEN": hf_token,  # D-213: empty; the user's token never enters a sandbox
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    }
    if agent_config is not None:
        env["AGENT_CONFIG"] = agent_config
    if agent_effort is not None:
        env["AGENT_EFFORT"] = agent_effort
    env["TIME_BUDGET_HOURS"] = redruntime
    env["AGENT_RESUME"] = "true" if resume_agent else "false"
    if seed is not None:
        env["TASK_SEED"] = str(seed)
        env["PYTHONHASHSEED"] = str(seed)
    env["PATH"] = AGENT_PATH
    env["BASH_MAX_TIMEOUT_MS"] = BASH_TIMEOUT_MS
    env["BASH_DEFAULT_TIMEOUT_MS"] = BASH_TIMEOUT_MS
    return env


def evaluator_env(*, seed: int | None = None, home: str = EVAL_HOME) -> dict[str, str]:
    """The variables upstream's evaluator container runs `evaluate.py` with (`:479-490`).

    D-092: only what `evaluate.py` uses; no API keys, as no scorer calls a paid model (D-215).
    `home` is `EVAL_HOME`, or `SHARED_EVAL_HOME` in a sandbox with no `evaluator` service.
    """
    env = {
        "HOME": home,
        "USER": AGENT_USER,
        "LOGNAME": AGENT_USER,
        "HF_HOME": HF_HOME,
        "PATH": IMAGE_PATH,
    }
    if seed is not None:
        env["TASK_SEED"] = str(seed)
        env["PYTHONHASHSEED"] = str(seed)
    return env
