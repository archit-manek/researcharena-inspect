"""Where the agent runs, as upstream's container sets it up (`src/pipeline_redteam.sh:357-378`)."""

AGENT_HOME = "/home/user"
"""`--home "$AGENT_JOB_DIR:/home/user"`."""

AGENT_TASK_DIR = "/home/user/task"
"""`--pwd "/home/user/task"`: the agent's working folder and the workspace it is scored on."""

AGENT_PATH = (
    "/home/user/task:/root/.local/bin:/usr/local/cuda/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
"""`--env PATH=...` (`:367`); `pre_agent_setup.sh` puts `/home/user/task` first again."""

CONTAINER_ENV = {
    "HOME": AGENT_HOME,
    "PATH": AGENT_PATH,
    "HF_HOME": "/home/user/hf_cache",
    "CUDA_VISIBLE_DEVICES": "0",
    "BASH_MAX_TIMEOUT_MS": "3600000",
    "BASH_DEFAULT_TIMEOUT_MS": "3600000",
}
"""The container-wide variables of `pipeline_redteam.sh:359-369` that hold no secret.

`HF_TOKEN` (`:360`) is the job's Hugging Face token and is left to the sandbox config. The
agent's own variables are in `agents.claude_code.claude_code_env`.
"""
