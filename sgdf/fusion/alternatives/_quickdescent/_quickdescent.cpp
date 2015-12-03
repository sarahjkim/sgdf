#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <array>
#include <cstdlib>
#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"

#ifdef _OPENMP
#include <omp.h>
#endif


class Quickdescent {
        PyArrayObject *arr_source, *arr_mask, *arr_tinyt, *arr_solution;
        size_t shape[2];
        bool* border_mask;
    public:
        Quickdescent() {};
        ~Quickdescent() {};
        int parseArgs(PyObject *args);
        int initializeGuess();
        float descend();
        inline bool isBorder(npy_intp y, npy_intp x);
    private:
        float averageBorderValue(PyArrayObject *im);
        float calcErrorForPixel(npy_intp y, npy_intp x);
        float calcErrorForPixelNeighbor(float solutionPixel, float sourcePixel,
                                        npy_intp y, npy_intp x);
};


/*
 * Returns whether or not the pixel at coordinates (y, x) in mask is a border
 * pixel.
 *
 * @param y     y-coordinate of pixel being considered
 * @param x     x-coordinate of pixel being considered
 * @return      Whether or not designated pixel in mask is a border pixel
 */
inline bool
Quickdescent::isBorder(npy_intp y, npy_intp x) {
    if (y < 0 || y >= shape[0] || x < 0 || x >= shape[1]) {
        return false;
    }

    bool isInner = *(bool *)PyArray_GETPTR2(arr_mask, y, x);
    if (isInner) {
        return false;
    }

    int neighbors = 0;
    if (y > 0 && *(bool *)PyArray_GETPTR2(arr_mask, y - 1, x)) {
        neighbors++;
    }
    if (y < shape[0] - 1 && *(bool *)PyArray_GETPTR2(arr_mask, y + 1, x)) {
        neighbors++;
    }
    if (x > 0 && *(bool *)PyArray_GETPTR2(arr_mask, y, x - 1)) {
        neighbors++;
    }
    if (x < shape[1] - 1 && *(bool *)PyArray_GETPTR2(arr_mask, y, x + 1)) {
        neighbors++;
    }
    return neighbors > 0;
}


/*
 * Parses arguments given to poisson_blend.
 *
 * @return       Designates successfully parsing the arguments
 * @side-effect  Sets the values of arr_source, arr_mask, arr_tinyt,
 *               arr_solution, and border_mask
 */
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

    // Extra error checking for Numpy input arrays
    std::array<PyArrayObject *, 4> ndarray_args = {{arr_source, arr_mask, arr_tinyt, arr_solution}};
    for (int i = 0; i < ndarray_args.size(); i++) {
        PyArrayObject *ndarray_arg = ndarray_args[i];
        if (ndarray_arg == NULL) {
            return 1;
        }
        if (PyArray_NDIM(ndarray_arg) != 2) {
            PyErr_SetString(PyExc_ValueError, "Input arrays should have dimension 2.");
            return 1;
        }

        npy_intp *dims = PyArray_DIMS(ndarray_arg);
        if (i == 0) {
            shape[0] = dims[0];
            shape[1] = dims[1];
        } else {
            if (shape[0] != dims[0] || shape[1] != dims[1]) {
                PyErr_Format(PyExc_ValueError, "Dimension mismatch: %ldx%ld and %zux%zu", dims[0], dims[1], shape[0], shape[1]);
                return 1;
            }
        }
    }

    // Set up border_mask
    npy_intp y, x;
    border_mask = new bool[shape[0] * shape[1]]();
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            if (isBorder(y, x)) {
                border_mask[y * shape[1] + x] = true;
            }
        }
    }

    return 0;
}


/*
 * Computes the starting point for gradient descent based on the following
 * heuristic. We start off the solution values to the source pixel values
 * plus the difference between the average border value of the target and
 * source images. This has the effect of bringing us closer to the final
 * pixel values by maintaining the gradients of the source image while bringing
 * the border values closer to the target/destination image.
 *
 * @return       Designates a successful initialization
 * @side-effect  Sets the values of arr_solution
 */
int
Quickdescent::initializeGuess() {
    float targetAvg = averageBorderValue(arr_tinyt);
    float sourceAvg = averageBorderValue(arr_source);
    float avgDiff = targetAvg - sourceAvg;

    npy_intp y, x;
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            float sourceVal = *(float *)PyArray_GETPTR2(arr_source, y, x);
            *(float*)PyArray_GETPTR2(arr_solution, y, x) = sourceVal + avgDiff;
        }
    }
    return 0;
}


/*
 * Completes one iteration of gradient descent:
 *   1. Calculates revised values of solution.
 *   2. Re-calculates error after revising solution.
 *
 * @return       Re-calculated error of revised solution
 * @side-effect  Sets arr_solution to new PyArrayObject with revised values
 */
float
Quickdescent::descend() {
    PyArrayObject *temp_solution; // TODO: Allocate new array for new values

    npy_intp y, x;

    #pragma omp parallel for
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            // TODO: Calculate new value at each pixel (y, x) and assign
            // to temp_solution[y][x]
        }
    }

    float error = 0.0;
    #pragma omp parallel for reduction(+:error)
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            error += calcErrorForPixel(y, x);
        }
    }
    return error;
}


PyObject *
poisson_blend(PyObject *self, PyObject *args) {
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
    // TODO: Shouldn't we also stop if the error is increasing?

    Py_RETURN_NONE;
}


/*
 * Averages the values of border pixels.
 *
 * @param im  2D array of pixel values (may be source or target image)
 * @return    Average of border values
 */
float
Quickdescent::averageBorderValue(PyArrayObject *im) {
    npy_intp y, x;

    float total = 0.0;
    int n = 0;
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            if (border_mask[y * shape[1] + x]) {
                total += *(float *)PyArray_GETPTR2(im, y, x);
                n += 1;
            }
        }
    }

    if (n == 0) return 0.0;
    else return total/n;
}


/*
 * Calculates error contributed by specified inner pixel i. For each neighbor j
 * of the specified pixel, add:
 *   - ((v_i - v_j) - (s_i - s_j)) ^ 2  if j is inner pixel
 *   - ((v_i - t_j) - (s_i - s_j)) ^ 2  if j is border pixel
 * where v_k is value of pixel k in the solution image,
 *       s_k is value of pixel k in the source image,
 *       t_k is value of pixel k in the target image.
 *
 * @param y  y-coordinate of inner pixel being considered
 * @param x  x-coordinate of inner pixel being considered
 * @return   Error contributed by pixel
 */
float
Quickdescent::calcErrorForPixel(npy_intp y, npy_intp x) {
    float totalError = 0.0;
    float solutionPixel = *(float *)PyArray_GETPTR2(arr_solution, y, x);
    float sourcePixel = *(float *)PyArray_GETPTR2(arr_source, y, x);
    totalError += calcErrorForPixelNeighbor(solutionPixel, sourcePixel,
                                            y - 1, x);
    totalError += calcErrorForPixelNeighbor(solutionPixel, sourcePixel,
                                            y + 1, x);
    totalError += calcErrorForPixelNeighbor(solutionPixel, sourcePixel,
                                            y, x - 1);
    totalError += calcErrorForPixelNeighbor(solutionPixel, sourcePixel,
                                            y, x + 1);
    return totalError;
}


/*
 * Calculates error contributed by gradients between inner pixel i and
 * neighbor pixel j.
 *   - ((v_i - v_j) - (s_i - s_j)) ^ 2  if j is inner pixel
 *   - ((v_i - t_j) - (s_i - s_j)) ^ 2  if j is border pixel
 * where v_k is value of pixel k in the solution image,
 *       s_k is value of pixel k in the source image,
 *       t_k is value of pixel k in the target image.
 *
 * @param solutionPixel  v_i
 * @param sourcePixel    s_i
 * @param y              y-coordinate of neighbor pixel being considered
 * @param x              x-coordinate of neighbor pixel being considered
 * @return               Error contributed by neighbor pixel
 */
float
Quickdescent::calcErrorForPixelNeighbor(float solutionPixel, float sourcePixel,
                                        npy_intp y, npy_intp x) {
    if (y < 0 || y >= shape[0] || x < 0 || x >= shape[1]) {
        return 0.0;
    }

    float sourceNeighborPixel = *(float *)PyArray_GETPTR2(arr_source, y, x);
    float otherNeighborPixel;

    if (*(bool *)PyArray_GETPTR2(arr_mask, y, x)) {
        otherNeighborPixel = *(float *)PyArray_GETPTR2(arr_solution, y, x);
    } else {
        otherNeighborPixel = *(float *)PyArray_GETPTR2(arr_tinyt, y, x);
    }

    float solutionDiff = solutionPixel - otherNeighborPixel;
    float sourceDiff = sourcePixel - sourceNeighborPixel;
    float error = solutionDiff - sourceDiff;
    return error * error;
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
