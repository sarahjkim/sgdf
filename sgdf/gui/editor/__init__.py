import logging
import matplotlib.pyplot as plt
import numpy as np
import Tkinter as tk
import tkFileDialog as filedialog
from collections import OrderedDict
from sgdf.fusion import get_fusion_algorithm
from sgdf.gui.editor.canvas import EditorViewCanvas
from sgdf.gui.editor.frame import EditorViewFrame
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
        self.root.geometry("1200x500")
        self.fusion = get_fusion_algorithm("reference")()
        self.mask_ndarray = None
        self.source_anchor = None
        self.target_anchor = None

    def handle_loadtarget(self, event=None, image_path=None):
        if not image_path:
            image_path = filedialog.askopenfilename()
        if image_path:
            _log.info("Loading target image file: %s", repr(image_path))
            im = plt.imread(image_path)
            h, w, channels = im.shape
            assert channels == 3, "TODO remove this"
            if np.max(im) > 1.0:
                im = im.astype(np.float32) / 255.
            self.fusion.set_target_image(im)
            self.mask_ndarray = np.zeros((h, w), dtype=np.bool)
            self.target_canvas.bind("<Button-1>", self.handle_brush_start)
            self.target_canvas.bind("<B1-Motion>", self.handle_brush_motion)
            self.target_canvas.draw_numpy(self.fusion.get_fusion())

    def handle_loadsource(self, event=None, image_path=None):
        if not image_path:
            image_path = filedialog.askopenfilename()
        if image_path:
            _log.info("Loading source image file: %s", repr(image_path))
            im = plt.imread(image_path)
            h, w, channels = im.shape
            assert channels == 3, "TODO remove this"
            if np.max(im) > 1.0:
                im = im.astype(np.float32) / 255.
            self.fusion.set_source_image(im)
            self.source_canvas.draw_numpy(im)
            self.source_canvas.bind("<Button-1>", self.handle_anchor)
            self.source_canvas.config(cursor="crosshair")

    def handle_brush_start(self, event):
        if self.target_anchor is None:
            self.target_anchor = np.array([event.y, event.x])
        if self.source_anchor is None:
            self.source_anchor = np.copy(self.target_anchor)
        self.fusion.set_anchor_points(self.source_anchor, self.target_anchor)
        self.handle_brush_motion(event)

    def handle_brush_motion(self, event):
        y_coord, x_coord = event.y, event.x
        t_height, t_width = self.mask_ndarray.shape
        if y_coord < t_height and x_coord < t_width:
            self.mask_ndarray[y_coord - 5 : y_coord + 5, x_coord - 5 : x_coord + 5] = True
            self.fusion.update_blend(self.mask_ndarray)
            self.target_canvas.draw_numpy(self.fusion.get_fusion())

    def handle_anchor(self, event):
        self.target_anchor = None
        self.source_anchor = np.array([event.y, event.x])
        self.fusion.commit_blend()
        self.mask_ndarray.fill(False)

    def handle_save(self, event=None):
        _log.info("Saving image file: %s", repr(filedialog.askopenfilename()))

    def handle_unimplemented(self, event=None):
        _log.warn("Unimplemented event handler triggered")

    def handle_quit(self, event=None):
        self.frame.quit()

    def mainloop(self):
        return self.root.mainloop()
