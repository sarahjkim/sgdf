import Tkinter as tk
import numpy as np


class PhotoImage(object):
    def __init__(self, ndarray=None, size=None, **kwargs):
        assert ndarray is not None or size is not None, "Please give me either a image or size"

        if ndarray is not None:
            size = ndarray.shape[:2]
        elif size is not None:
            ndarray = np.zeros(size)
        self.__nparray = ndarray

        kwargs["height"], kwargs["width"] = size[:2]
        self.__size = self.__nparray.shape
        self.__photo = tk.PhotoImage(**kwargs)
        self.tk = self.__photo.tk
        self.paste(ndarray)

    def __str__(self):
        """
        Get the Tkinter photo image identifier.  This method is automatically called by Tkinter
        whenever a PhotoImage object is passed to a Tkinter method.

        """
        return str(self.__photo)

    def width(self):
        return self.__size[0]

    def height(self):
        return self.__size[1]

    def paste(self, ndarray):
        interp = self.__photo.tk
        try:
            interp.call("PyNumpyTkPut", self.__photo, id(ndarray))
        except tk.TclError:
            # activate Tkinter hook
            try:
                from sgdf.gui import _numpytk
                try:
                    _numpytk.tkinit(interp.interpaddr(), 1)
                except AttributeError:
                    _numpytk.tkinit(id(interp), 0)
                interp.call("PyNumpyTkPut", self.__photo, id(ndarray))
            except (ImportError, AttributeError, tk.TclError):
                raise  # configuration problem; cannot attach to Tkinter
