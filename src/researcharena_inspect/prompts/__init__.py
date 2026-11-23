"""Red-team agent and monitor prompts, byte-identical to upstream's.

Every builder reads upstream's text files from the pinned checkout and is tested against
upstream's own scripts.
"""

from researcharena_inspect.prompts.agent import (
    AGENT_FAMILIES,
    AgentFamily,
    agent_instruction_file,
    agent_prompt,
    budget_seconds,
    build_prompt,
    format_hours,
    resume_prompt,
)
from researcharena_inspect.prompts.monitor import (
    ACCESS_TYPES,
    INSTRUCTION_FILENAME,
    TRANSCRIPT_FILENAME,
    MonitorAccess,
    build_monitor_prompt,
    list_directory,
    monitor_prompt,
    monitor_prompt_file,
    monitor_task_instruction,
)
from researcharena_inspect.prompts.settings import (
    BASELINE,
    default_task_config,
    list_settings,
    list_side_tasks,
    resume_agent,
    task_configs,
)

__all__ = [
    "ACCESS_TYPES",
    "AGENT_FAMILIES",
    "BASELINE",
    "INSTRUCTION_FILENAME",
    "TRANSCRIPT_FILENAME",
    "AgentFamily",
    "MonitorAccess",
    "agent_instruction_file",
    "agent_prompt",
    "budget_seconds",
    "build_monitor_prompt",
    "build_prompt",
    "default_task_config",
    "format_hours",
    "list_directory",
    "list_settings",
    "list_side_tasks",
    "monitor_prompt",
    "monitor_prompt_file",
    "monitor_task_instruction",
    "resume_agent",
    "resume_prompt",
    "task_configs",
]
