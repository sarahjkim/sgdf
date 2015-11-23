#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <cstdlib>
#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"


class Quickdescent {
        PyArrayObject *arr_source, *arr_mask, *arr_tinyt, *arr_solution;
    public:
        Quickdescent() {};
        ~Quickdescent() {};
        int parseArgs(PyObject *args);
        int initializeGuess();
        float descend();
};


int
Quickdescent::parseArgs(PyObject *args) {
    PyObject *arg_source, *arg_mask, *arg_tinyt, *arg_solution;
    if (!PyArg_ParseTuple(args, "OOOO!", &arg_source, &arg_mask, &arg_tinyt, &PyArray_Type, &arg_solution)) {
        return 1;
    }

    arr_source = (PyArrayObject *) PyArray_FROM_OTF(arg_source, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (arr_source == NULL) {
        return 1;
    }

    arr_mask = (PyArrayObject *) PyArray_FROM_OTF(arg_mask, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (arr_mask == NULL) {
        return 1;
    }

    arr_tinyt = (PyArrayObject *) PyArray_FROM_OTF(arg_tinyt, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (arr_tinyt == NULL) {
        return 1;
    }

    arr_solution = (PyArrayObject *) PyArray_FROM_OTF(arg_solution, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);
    if (arr_solution == NULL) {
        return 1;
    }

    return 0;
}


static inline bool isBorder(PyArrayObject *mask, npy_intp y, npy_intp x) {
    npy_intp *dims = PyArray_DIMS(mask);
    if (y == 0 || y == dims[0]-1 || x == 0 || x == dims[1]-1) {
        return true;
    }
    return !(*(bool *)PyArray_GETPTR2(mask, y - 1, x) &&
             *(bool *)PyArray_GETPTR2(mask, y + 1, x) &&
             *(bool *)PyArray_GETPTR2(mask, y, x - 1) &&
             *(bool *)PyArray_GETPTR2(mask, y, x + 1));
}


static float averageBorderValue(PyArrayObject *im, PyArrayObject *mask) {
    npy_intp y, x;
    npy_intp *dims = PyArray_DIMS(mask);
    float total = 0.0;
    int n = 0;
    for (y = 0; y < dims[0]; y++) {
        for (x = 0; x < dims[1]; x++) {
            if (isBorder(mask, y, x)) {
                total += *(float *)PyArray_GETPTR2(mask, y, x);
                n += 1;
            }
        }
    }
    if (n == 0) return 0.0;
    else return total/n;
}


int
Quickdescent::initializeGuess() {
    npy_intp y, x;
    npy_intp *dims = PyArray_DIMS(arr_mask);
    for (y = 0; y < dims[0]; y++) {
        for (x = 0; x < dims[1]; x++) {
            // TODO this is not the code we need
            *(float*)PyArray_GETPTR2(arr_solution, y, x) = *(float*)PyArray_GETPTR2(arr_source, y, x);
        }
    }
}


PyObject *poisson_blend(PyObject *self, PyObject *args) {
    Quickdescent quickdescent;

    if (quickdescent.parseArgs(args))
        return NULL;

    if (quickdescent.initializeGuess())
        return NULL;

    const float epsilon = 0.0001;
    float error = quickdescent.descend(), previous_error, delta_error;
    do {
        previous_error = error;
        error = quickdescent.descend();
        delta_error = error - previous_error;
    } while (delta_error > epsilon || delta_error < -epsilon);

    Py_RETURN_NONE;
}


static PyMethodDef functions[] = {
    {"poisson_blend", (PyCFunction) poisson_blend, METH_VARARGS},
    {NULL, NULL}
};


PyMODINIT_FUNC
init_quickdescent(void) {
    Py_InitModule("_quickdescent", functions);
    import_array();
}
