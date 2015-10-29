import Tkinter as tk
import traceback


class EditorView(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Streaming Gradient Domain Fusion")
        self.frame = tk.Frame(self.root)
        self.canvas = tk.Canvas(self.frame, background="#009900", height=500, width=600)

        def click_handler(event):
            traceback.print_stack()

        block = self.canvas.create_rectangle(0, 0, 100, 100, fill="#990000", outline="")
        self.canvas.tag_bind(block, "<Button-1>", click_handler)
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.frame.pack()

    def mainloop(self):
        return self.root.mainloop()
