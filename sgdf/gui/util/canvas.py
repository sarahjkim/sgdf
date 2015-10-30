import Tkinter as tk


class ResizingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        """
        A subclass of tk.Canvas that deals with resizing.

        Adapted from http://stackoverflow.com/questions/22835289/

        """
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.config(width=self.width, height=self.height)
