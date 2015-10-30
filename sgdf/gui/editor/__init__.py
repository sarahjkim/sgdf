import matplotlib.pyplot as plt
import Tkinter as tk
import tkFileDialog as filedialog
import traceback
from PIL import Image, ImageTk
from sgdf.gui.util.canvas import ResizingCanvas
from sgdf.gui.util.keyboard import SUPER
from sgdf.gui.util.menu import MenuBuilder


class EditorView(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Streaming Gradient Domain Fusion")
        self.frame = tk.Frame(self.root)
        self.canvas = ResizingCanvas(self.frame, background="#009900", height=500, width=600,
                                     highlightthickness=0)
        menu_items = {"File": [("Load image", self.handle_loadimage, (SUPER, "o")),
                               ("Save canvas", self.handle_savecanvas, (SUPER, "s")),
                               (),
                               ("Close", self.handle_quit, (SUPER, "w"))]}
        self.menu = MenuBuilder(items=menu_items)
        self.menu.render(self.root)
        self.frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

    def handle_loadimage(self, event=None):
        image_path = filedialog.askopenfilename()
        if image_path:
            # TODO remove
            print "Loading file:", repr(image_path)
            im = plt.imread(image_path)
            h, w, channels = im.shape
            assert channels == 3, "TODO remove this"
            self.tk_image = ImageTk.PhotoImage(Image.fromarray(im))
            self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

    def handle_savecanvas(self, event=None):
        # TODO replace
        print "Saved file:", repr(filedialog.askopenfilename())

    def handle_quit(self, event=None):
        self.frame.quit()

    def mainloop(self):
        return self.root.mainloop()
