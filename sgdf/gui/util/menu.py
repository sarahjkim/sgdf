import Tkinter as tk
from collections import OrderedDict
from sgdf.gui.util.keyboard import get_visual_keybinding, get_symbolic_keybinding


class MenuBuilder(object):
    def __init__(self, **kwargs):
        """
        A utility class for building window menus and managing keyboard shortcuts via Tkinter.

        ARGUMENTS

            items -- A OrderedDict of menu category (string) to menu items (list), where the menu
                     items are tuples of (label, command, accelerator, underline).  You can insert
                     seperators by including an empty tuple (all arguments are optional). See the
                     render() method for more information.

        """
        self.items = OrderedDict()
        if "items" in kwargs:
            self.extend(kwargs["items"])

    def extend(self, items):
        self.items.update(items)

    def render(self, root):
        def render_menuitem(menu, *args):
            keys = ["label", "command", "accelerator", "underline"]
            args = list(args) + [None] * (len(keys) - len(args))
            kwargs = dict(zip(keys, args))
            if kwargs["label"]:
                if kwargs["accelerator"]:
                    # Also add keyboard shortcuts to application root
                    root.bind_all(get_symbolic_keybinding(kwargs["accelerator"]), kwargs["command"])
                    kwargs["accelerator"] = get_visual_keybinding(kwargs["accelerator"])
                menu.add_command(**kwargs)
            else:
                menu.add_separator()

        menubar = tk.Menu(root)
        for category, category_items in self.items.items():
            menu = tk.Menu(menubar, tearoff=0)
            map(lambda category_item: render_menuitem(menu, *category_item), category_items)
            menubar.add_cascade(label=category, menu=menu)
        root.config(menu=menubar)

    def clone(self):
        return MenuBuilder(items=self.items)
