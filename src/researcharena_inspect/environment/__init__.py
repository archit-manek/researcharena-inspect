"""The environment the red-team agent works in: image, sandbox, workspace staging.

- `image`: the image built from `Dockerfile` (upstream's `containers/arena.def`), by digest.
- `sandbox`: `sandbox_spec()`, the Inspect sandbox: Docker with one GPU.
- `layout`: paths in the sandbox, the agent's user and its environment variables.
- `staging`: `stage_workspace` and the setup solver `stage_agent_workspace`, which put into
  the sandbox what upstream's `pipeline_redteam.sh` puts in the agent's home.
- `evaluation`: `prepare_evaluation`, what upstream does between the agent and `evaluate.py`,
  and `evaluator_sandbox`, the evaluator's own container on Docker (D-093).
- `check`: `environment_check`, a task that checks all of it on a GPU without a model.
"""

from researcharena_inspect.environment.evaluation import (
    EVAL_ATTEMPTS,
    FAILED_METRICS,
    METRICS_PATH,
    SNAPSHOT_DIR,
    SUBMISSION_DIR,
    EvaluationCommand,
    evaluator_entries,
    evaluator_sandbox,
    prepare_evaluation,
    stop_agent_container,
    stop_agent_processes,
)
from researcharena_inspect.environment.image import IMAGE, IMAGE_DIGEST, IMAGE_REPOSITORY
from researcharena_inspect.environment.layout import (
    AGENT_HOME,
    AGENT_PATH,
    AGENT_UID,
    AGENT_USER,
    EVAL_DIR,
    EVAL_HOME,
    EVALUATOR_SERVICE,
    HF_HOME,
    PRE_AGENT_SETUP,
    PROMPT_FILE,
    SHARED_EVAL_HOME,
    TASK_DIR,
    agent_env,
    evaluator_env,
)
from researcharena_inspect.environment.sandbox import (
    COMPOSE_FILE,
    sandbox_spec,
    with_evaluator,
)
from researcharena_inspect.environment.staging import (
    Entry,
    StagingError,
    StagingResult,
    home_entries,
    stage_agent_workspace,
    stage_workspace,
)

__all__ = [
    "AGENT_HOME",
    "AGENT_PATH",
    "AGENT_UID",
    "AGENT_USER",
    "COMPOSE_FILE",
    "EVALUATOR_SERVICE",
    "EVAL_ATTEMPTS",
    "EVAL_DIR",
    "EVAL_HOME",
    "FAILED_METRICS",
    "HF_HOME",
    "IMAGE",
    "IMAGE_DIGEST",
    "IMAGE_REPOSITORY",
    "METRICS_PATH",
    "PRE_AGENT_SETUP",
    "PROMPT_FILE",
    "SHARED_EVAL_HOME",
    "SNAPSHOT_DIR",
    "SUBMISSION_DIR",
    "TASK_DIR",
    "Entry",
    "EvaluationCommand",
    "StagingError",
    "StagingResult",
    "agent_env",
    "evaluator_entries",
    "evaluator_env",
    "evaluator_sandbox",
    "home_entries",
    "prepare_evaluation",
    "sandbox_spec",
    "stage_agent_workspace",
    "stage_workspace",
    "stop_agent_container",
    "stop_agent_processes",
    "with_evaluator",
]
