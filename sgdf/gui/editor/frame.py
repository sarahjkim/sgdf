import Tkinter as tk


class EditorViewFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        """
        A subclass of tk.Frame that contains several canvas elements. This frame will dynamically
        resize the canvas elements when the frame itself is resized, so that the frame width is
        equally distributed between them.

        """
        tk.Frame.__init__(self, *args, **kwargs)
        self.canvas_set = []
        self.bind("<Configure>", self.on_resize)

    def register_canvas_set(self, *args):
        self.canvas_set = args

    def on_resize(self, event):
        assert self.canvas_set
        height = event.height
        width = event.width
        width_per_canvas = max(0, width / len(self.canvas_set))
        width_remainder = width - width_per_canvas * (len(self.canvas_set) - 1)
        for canvas in self.canvas_set[:-1]:
            canvas.config(height=height, width=width_per_canvas)
        self.canvas_set[-1].config(height=height, width=width_remainder)
