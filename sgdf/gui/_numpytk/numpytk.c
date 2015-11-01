#include <stdlib.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "tk.h"
#include "tk.h"

/* copied from _tkinter.c (this isn't as bad as it may seem: for new
   versions, we use _tkinter's interpaddr hook instead, and all older
   versions use this structure layout) */

typedef struct {
    PyObject_HEAD
    Tcl_Interp* interp;
} TkappObject;


static PyArrayObject *lookup_ndarray(const char *id) {
    return (PyArrayObject *) atol(id);
}

static int PyNumpyTkPut(ClientData clientdata, Tcl_Interp* interp, int argc, const char **argv) {
    Tk_PhotoImageBlock block;
    Tk_PhotoHandle photo;
    PyArrayObject *ndarray;

    if (argc != 3) {
        Tcl_AppendResult(interp, "usage: ", argv[0],
                         " destPhoto srcNdarray", (char *) NULL);
        return TCL_ERROR;
    }

    photo = Tk_FindPhoto(interp, argv[1]);
    if (photo == NULL) {
        Tcl_AppendResult(interp, "destination photo must exist", (char *) NULL);
        return TCL_ERROR;
    }

    ndarray = lookup_ndarray(argv[2]);

    if (PyArray_NDIM(ndarray) != 3) {
        Tcl_AppendResult(interp, "ndarray dimensions should be 3", (char *) NULL);
        return TCL_ERROR;
    }

    npy_intp *ndarray_dims = PyArray_DIMS(ndarray);
    if (ndarray_dims[2] != 3) {
        Tcl_AppendResult(interp, "ndarray 3rd dimension should have size 3", (char *) NULL);
        return TCL_ERROR;
    }

    npy_intp *ndarray_strides = PyArray_STRIDES(ndarray);

    block.pixelSize = ndarray_strides[1];
    block.offset[0] = 0;
    block.offset[1] = 1;
    block.offset[2] = 2;
    block.offset[3] = 0;
    block.height = ndarray_dims[0];
    block.width = ndarray_dims[1];
    block.pitch = ndarray_strides[0];
    block.pixelPtr = (unsigned char *) PyArray_BYTES(ndarray);

    return Tk_PhotoPutBlock(interp, photo, &block, 0, 0, block.width, block.height,
            TK_PHOTO_COMPOSITE_SET);
}

static PyObject *_tkinit(PyObject *self, PyObject *args) {
    Tcl_Interp *interp;

    Py_ssize_t arg;
    int is_interp;
    if (!PyArg_ParseTuple(args, "ni", &arg, &is_interp))
        return NULL;

    if (is_interp)
        interp = (Tcl_Interp *) arg;
    else {
        TkappObject *app;
        /* Do it the hard way.  This will break if the TkappObject
           layout changes */
        app = (TkappObject *) arg;
        interp = app->interp;
    }

    Tcl_CreateCommand(interp, "PyNumpyTkPut", &PyNumpyTkPut, (ClientData) 0, (Tcl_CmdDeleteProc*) NULL);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef functions[] = {
    {"tkinit", (PyCFunction)_tkinit, 1},
    {NULL, NULL}
};

PyMODINIT_FUNC
init_numpytk(void) {
    Py_InitModule("_numpytk", functions);
}
