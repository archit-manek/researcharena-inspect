"""What upstream's bash does to a prompt on its way to a CLI, for byte-for-byte comparison."""


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
