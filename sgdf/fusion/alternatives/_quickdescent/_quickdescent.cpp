#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <cstdlib>
#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"


static inline bool isBorder(PyArrayObject *mask, npy_intp y, npy_intp x);

class Quickdescent {
        PyArrayObject *arr_source, *arr_mask, *arr_tinyt, *arr_solution;
        bool** border_mask;
    public:
        Quickdescent() {};
        ~Quickdescent() {};
        int parseArgs(PyObject *args);
        float averageBorderValue(PyArrayObject *im);
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

    // Set up border_mask
    npy_intp y, x;
    npy_intp *dims = PyArray_DIMS(arr_mask);
    border_mask = new bool *[dims[0]];
    for (int i = 0; i < dims[0]; i++) {
        border_mask[i] = new bool[dims[1]]();
    }
    for (y = 0; y < dims[0]; y++) {
        for (x = 0; x < dims[1]; x++) {
            if (isBorder(arr_mask, y, x)) {
                border_mask[y][x] = true;
            }
        }
    }

    return 0;
}


/*
 * Returns whether or not the pixel at coordinates (y, x) in mask is a border
 * pixel.
 *
 * @param mask Designates inner pixels (values calculated based on
 *             source/target values)
 * @param y    y-coordinate of pixel being considered
 * @param x    x-coordinate of pixel being considered
 * @return     Whether or not designated pixel in mask is a border pixel
 */
static inline bool isBorder(PyArrayObject *mask, npy_intp y, npy_intp x) {
    npy_intp *dims = PyArray_DIMS(mask);
    if (y < 0 || y >= dims[0] || x < 0 || x >= dims[1]) {
        return false;
    }

    bool isInner = *(bool *)PyArray_GETPTR2(mask, y, x);
    if (isInner) {
        return false;
    }

    int neighbors = 0;
    if (y > 0 && *(bool *)PyArray_GETPTR2(mask, y - 1, x)) {
        neighbors++;
    }
    if (y < dims[0] - 1 && *(bool *)PyArray_GETPTR2(mask, y + 1, x)) {
        neighbors++;
    }
    if (x > 0 && *(bool *)PyArray_GETPTR2(mask, y, x - 1)) {
        neighbors++;
    }
    if (x < dims[1] - 1 && *(bool *)PyArray_GETPTR2(mask, y, x + 1)) {
        neighbors++;
    }
    return neighbors > 0;
}


/*
 * Averages the values of border pixels.
 *
 * @param im 2D array of pixel values (may be source or target image)
 * @return   Average of border values
 */
float
Quickdescent::averageBorderValue(PyArrayObject *im) {
    npy_intp y, x;
    npy_intp *dims = PyArray_DIMS(arr_mask);

    float total = 0.0;
    int n = 0;
    for (y = 0; y < dims[0]; y++) {
        for (x = 0; x < dims[1]; x++) {
            if (border_mask[y][x]) {
                total += *(float *)PyArray_GETPTR2(arr_mask, y, x);
                n += 1;
            }
        }
    }

    if (n == 0) return 0.0;
    else return total/n;
}


/*
 * Computes the starting point for gradient descent based on the following
 * heuristic. We start off the solution values to the source pixel values
 * plus the difference between the average border value of the target and
 * source images. This has the effect of bringing us closer to the final
 * pixel values by maintaining the gradients of the source image while bringing
 * the border values closer to the target/destination image.
 *
 * @return      Designates a successful initialization
 * @side-effect Sets the values of arr_solution
 */
int
Quickdescent::initializeGuess() {
    float targetAvg = averageBorderValue(arr_tinyt);
    float sourceAvg = averageBorderValue(arr_source);
    float avgDiff = targetAvg - sourceAvg;

    npy_intp y, x;
    npy_intp *dims = PyArray_DIMS(arr_mask);
    for (y = 0; y < dims[0]; y++) {
        for (x = 0; x < dims[1]; x++) {
            float sourceVal = *(float *)PyArray_GETPTR2(arr_source, y, x);
            *(float*)PyArray_GETPTR2(arr_solution, y, x) = sourceVal + avgDiff;
        }
    }
    return 0;
}


PyObject *poisson_blend(PyObject *self, PyObject *args) {
    Quickdescent quickdescent;

    if (!quickdescent.parseArgs(args))
        Py_RETURN_NONE;

    if (!quickdescent.initializeGuess())
        Py_RETURN_NONE;

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
