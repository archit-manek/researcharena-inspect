"""What upstream's shell glue does to text on its way to a CLI.

Upstream moves every prompt through bash: `X=$(python3 builder.py ...)` keeps a script's
stdout without its trailing newlines, and `echo "$X" > file` writes it back with one. These
two helpers reproduce that so the port's output can be compared byte for byte.
"""


def command_substitution(output: str) -> str:
    """What `$(...)` keeps of a command's output: all of it except trailing newlines.

    Bash also drops NUL bytes there; prompt text never contains them.
    """
    return output.rstrip("\n")


def echo_line(text: str) -> str:
    """What `echo "$X"` writes: the text and one newline (bash's builtin echo, no options)."""
    return text + "\n"


def script_output(text: str) -> str:
    """What `print(text)` in one of upstream's builder scripts writes to stdout."""
    return text + "\n"
