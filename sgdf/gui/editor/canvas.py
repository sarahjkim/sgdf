import Tkinter as tk
from sgdf.benchmarking import log_timer
from sgdf.gui.numpytk import PhotoImage


class EditorViewCanvas(tk.Canvas):
    def __init__(self, *args, **kwargs):
        tk.Canvas.__init__(self, *args, **kwargs)
        self.active_image_id = self.create_image(0, 0, anchor=tk.NW)
        self.active_image_container = None

    def draw_numpy(self, ndarray):
        """
        Draws a numpy array (H * W * 3) as an image on this canvas.

        """
        with log_timer("EditorViewCanvas.draw_numpy"):
            if self.active_image_container is None:
                self.active_image_container = PhotoImage(ndarray=ndarray)
                self.itemconfig(self.active_image_id, image=self.active_image_container)
            else:
                self.active_image_container.paste(ndarray=ndarray)
