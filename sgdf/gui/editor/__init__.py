import logging
import matplotlib.pyplot as plt
import Tkinter as tk
import tkFileDialog as filedialog
from collections import OrderedDict
from PIL import Image, ImageTk
from sgdf.benchmarking import log_timer
from sgdf.gui.util.keyboard import SUPER, SHIFT, CONTROL
from sgdf.gui.util.menu import MenuBuilder


_log = logging.getLogger(__name__)

class EditorView(object):
    def __init__(self):
        """This is the View that renders the main editor window."""
        self.root = tk.Tk()
        self.root.wm_title("Streaming Gradient Domain Fusion")
        self.frame = EditorViewFrame(self.root)

        # Set up editor canvas
        self.target_canvas = EditorViewCanvas(self.frame, background="#009900",
                                              highlightthickness=0)
        self.target_canvas.bind("<Button-1>", self.handle_loadtarget)
        self.source_canvas = EditorViewCanvas(self.frame, background="#00477d",
                                              highlightthickness=0)
        self.source_canvas.bind("<Button-1>", self.handle_loadsource)
        self.frame.register_canvas_set(self.target_canvas, self.source_canvas)
        self.target_canvas.grid(row=0, column=1)
        self.source_canvas.grid(row=0, column=2)

        # Set up editor menu
        menu_items = OrderedDict()
        menu_items["File"] = [("Load target image", self.handle_loadtarget, (SUPER, "o")),
                              ("Load source image", self.handle_loadsource, (SUPER, SHIFT, "o")),
                              ("Save canvas", self.handle_save, (SUPER, "s")),
                              (),
                              ("Close", self.handle_quit, (SUPER, "w"))]
        menu_items["Edit"] = [("Undo", self.handle_unimplemented, (SUPER, "z"))]
        menu_items["Brush"] = [("Circle (feathered)", self.handle_unimplemented, (SUPER, "1")),
                               ("Circle", self.handle_unimplemented, (SUPER, "2")),
                               ("Square (feathered)", self.handle_unimplemented, (SUPER, "3")),
                               ("Square", self.handle_unimplemented, (SUPER, "4")),
                               (),
                               ("Increase brush size", self.handle_unimplemented, (CONTROL, "]")),
                               ("Decrease brush size", self.handle_unimplemented, (CONTROL, "["))]
        menu_items["Help"] = []
        self.menu = MenuBuilder(items=menu_items)
        self.menu.render(self.root)

        # Set up editor window
        self.frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.root.geometry("1232x500")

    def handle_loadtarget(self, event=None):
        image_path = filedialog.askopenfilename()
        if image_path:
            _log.info("Loading target image file: %s", repr(image_path))
            im = plt.imread(image_path)
            h, w, channels = im.shape
            assert channels == 3, "TODO remove this"
            self.target_canvas.draw_numpy(im)

    def handle_loadsource(self, event=None):
        image_path = filedialog.askopenfilename()
        if image_path:
            _log.info("Loading source image file: %s", repr(image_path))
            im = plt.imread(image_path)
            h, w, channels = im.shape
            assert channels == 3, "TODO remove this"
            self.source_canvas.draw_numpy(im)

    def handle_save(self, event=None):
        _log.info("Saving image file: %s", repr(filedialog.askopenfilename()))

    def handle_unimplemented(self, event=None):
        _log.warn("Unimplemented event handler triggered")

    def handle_quit(self, event=None):
        self.frame.quit()

    def mainloop(self):
        return self.root.mainloop()


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
                self.active_image_container = ImageTk.PhotoImage(Image.fromarray(ndarray))
                self.itemconfig(self.active_image_id, image=self.active_image_container)
            else:
                self.active_image_container.paste(Image.fromarray(ndarray))
        self.config(cursor="circle")
        self.unbind("<Button-1>")
