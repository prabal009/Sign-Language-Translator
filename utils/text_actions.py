"""Pure text operations shared by Qt manual and webcam edit paths."""

from string import ascii_uppercase


def apply_action(text: str, action: str) -> str:
    if len(action) == 1 and action in ascii_uppercase + "01":
        return text + action
    if action == "SPACE":
        return text + " " if text and not text[-1].isspace() else text
    if action == "BACKSPACE":
        return text[:-1]
    if action == "CLEAR_WORD":
        return text.rstrip().rpartition(" ")[0].rstrip()
    if action == "CLEAR":
        return ""
    return text
