import sys

SUPER = 1
SHIFT = 2
CONTROL = 3
_VISUAL = {SUPER: "Command" if sys.platform == "darwin" else "Control",
           SHIFT: "Shift",
           CONTROL: "Control"}
_SYMBOLIC = {SUPER: "Command" if sys.platform == "darwin" else "Control",
             SHIFT: "Shift",
             CONTROL: "Control"}


def get_visual_keybinding(binding):
    return "+".join([_VISUAL.get(part, part) for part in binding])


def get_symbolic_keybinding(binding):
    return "<%s>" % ("-".join([_SYMBOLIC.get(part, part) for part in binding]))
