import sys

SUPER = 1
_VISUAL = {SUPER: "Command" if sys.platform == "darwin" else "Control"}
_SYMBOLIC = {SUPER: "Command" if sys.platform == "darwin" else "Control"}


def get_visual_keybinding(binding):
    return "+".join([_VISUAL.get(part, part) for part in binding])


def get_symbolic_keybinding(binding):
    return "<%s>" % ("-".join([_SYMBOLIC.get(part, part) for part in binding]))
