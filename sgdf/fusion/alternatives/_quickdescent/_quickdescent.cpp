#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"

#ifdef _OPENMP
#include <omp.h>
#endif


class Quickdescent {
        PyArrayObject *arr_source, *arr_mask, *arr_tinyt, *arr_solution, *arr_scratch;
        size_t shape[2];

        float epsilon;
        int max_iterations;
        PyArrayObject *arr_errorlog;
    public:
        Quickdescent() {};
        ~Quickdescent() {};
        int parseArgs(PyObject *args);
        int initializeGuess();
        int blend();
    private:
        float averageBorderValue(PyArrayObject *im);
        float descend(float learning_rate);
        inline bool isBorder(npy_intp y, npy_intp x);
        inline float getSource(const npy_intp y, const npy_intp x);
        inline bool getMask(const npy_intp y, const npy_intp x);
        inline float getTarget(const npy_intp y, const npy_intp x);
        inline float &getSolution(const npy_intp y, const npy_intp x);
        inline float &getScratch(const npy_intp y, const npy_intp x);
        inline float &getErrorlog(const npy_intp n);
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
    assert(y < 0 || y >= shape[0] || x < 0 || x >= shape[1]);

    bool isInner = getMask(y, x);
    if (isInner) {
        return false;
    }

    int neighbors = 0;
    if (y > 0 && getMask(y - 1, x)) {
        neighbors++;
    }
    if (y < shape[0] - 1 && getMask(y + 1, x)) {
        neighbors++;
    }
    if (x > 0 && getMask(y, x - 1)) {
        neighbors++;
    }
    if (x < shape[1] - 1 && getMask(y, x + 1)) {
        neighbors++;
    }
    return neighbors > 0;
}


/* Returns a pixel in arr_source. */
inline float
Quickdescent::getSource(const npy_intp y, const npy_intp x) {
    return *(float *)PyArray_GETPTR2(arr_source, y, x);
}


/* Returns a pixel in arr_mask. */
inline bool
Quickdescent::getMask(const npy_intp y, const npy_intp x) {
    return *(bool *)PyArray_GETPTR2(arr_mask, y, x);
}


/* Returns a pixel in arr_tinyt. */
inline float
Quickdescent::getTarget(const npy_intp y, const npy_intp x) {
    return *(float *)PyArray_GETPTR2(arr_tinyt, y, x);
}


/* Returns a reference to a pixel in arr_solution. */
inline float &
Quickdescent::getSolution(const npy_intp y, const npy_intp x) {
    return *(float *)PyArray_GETPTR2(arr_solution, y, x);
}


/* Returns a reference to a pixel in arr_scratch. */
inline float &
Quickdescent::getScratch(const npy_intp y, const npy_intp x) {
    return *(float *)PyArray_GETPTR2(arr_scratch, y, x);
}


/* Returns a reference to a pixel in arr_errorlog. */
inline float &
Quickdescent::getErrorlog(const npy_intp n) {
    return *(float *)PyArray_GETPTR1(arr_errorlog, n);
}


/*
 * Parses arguments given to poisson_blend.
 *
 * @return       Designates successfully parsing the arguments
 * @side-effect  Sets up the parameters of Quickdescent.
 */
int
Quickdescent::parseArgs(PyObject *args) {
    PyObject *arg_source, *arg_mask, *arg_tinyt, *arg_solution, *arg_errorlog, *arg_scratch;
    if (!PyArg_ParseTuple(args, "OOOO!O!O!fi", &arg_source, &arg_mask, &arg_tinyt,
                          &PyArray_Type, &arg_solution, &PyArray_Type, &arg_scratch,
                          &PyArray_Type, &arg_errorlog, &epsilon, &max_iterations)) {
        return 1;
    }

    arr_source = (PyArrayObject *) PyArray_FROM_OTF(arg_source, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    arr_mask = (PyArrayObject *) PyArray_FROM_OTF(arg_mask, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    arr_tinyt = (PyArrayObject *) PyArray_FROM_OTF(arg_tinyt, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    arr_solution = (PyArrayObject *) PyArray_FROM_OTF(arg_solution, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);
    arr_scratch = (PyArrayObject *) PyArray_FROM_OTF(arg_scratch, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);

    // Extra error checking for Numpy input arrays
    std::array<PyArrayObject *, 5> ndarray_args = {{arr_source,
                                                    arr_mask,
                                                    arr_tinyt,
                                                    arr_solution,
                                                    arr_scratch}};
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

    {
        arr_errorlog = (PyArrayObject *) PyArray_FROM_OTF(arg_errorlog, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);
        if (arr_errorlog == NULL) {
            return 1;
        }
        if (PyArray_NDIM(arr_errorlog) != 1) {
            PyErr_SetString(PyExc_ValueError, "Error log should be a 1 dimensional array.");
            return 1;
        }

        npy_intp *dims = PyArray_DIMS(arr_errorlog);
        if (max_iterations == 0) {
            max_iterations = dims[0];
        } else if (max_iterations != dims[0]) {
            PyErr_SetString(PyExc_ValueError, "Error log should be the same size as max_iterations.");
            return 1;
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
    float average_target = averageBorderValue(arr_tinyt);
    float average_source = averageBorderValue(arr_source);
    float average_delta = average_target - average_source;

    npy_intp y, x;
    float pixel_value;
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            if (getMask(y, x)) {
                pixel_value = getSource(y, x) + average_delta;
                if (pixel_value < 0.0) pixel_value = 0.0;
                if (pixel_value > 1.0) pixel_value = 1.0;
                getSolution(y, x) = pixel_value;
            } else {
                getSolution(y, x) = getTarget(y, x);
            }
        }
    }
    return 0;
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

    double total = 0.0;
    int n = 0;
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            if (isBorder(y, x)) {
                total += *(float *)PyArray_GETPTR2(im, y, x);
                n += 1;
            }
        }
    }

    if (n == 0) return 0.0;
    else return (float) total/n;
}


int
Quickdescent::blend() {
    float error, previous_error, delta_error;
    float learning_rate;
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        learning_rate = 5.0 * max_iterations / (max_iterations + iteration);
        error = descend(learning_rate);
        getErrorlog(iteration) = error;
        if (iteration > 0) {
            delta_error = error - previous_error;
            if (fabs(delta_error) <= epsilon)
                break;
        }
        previous_error = error;
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
Quickdescent::descend(float learning_rate) {
    npy_intp y, x;
    double total_error = 0, total_dfdp = 0;

    #pragma omp parallel for reduction(+:total_error)
    for (y = 0; y < shape[0]; y++) {
        for (x = 0; x < shape[1]; x++) {
            double dfdp = 0;

            #define append_error(e) do { \
                double e_ = (e); \
                dfdp += e_; \
                total_error += e_ * e_; \
            } while (0)

            if (getMask(y, x)) {
                float solution_p = getSolution(y, x);
                float source_p = getSource(y, x);
                if (y > 0) {
                    append_error((solution_p - getSolution(y - 1, x)) - (source_p - getSource(y - 1, x)));
                }
                if (x > 0) {
                    append_error((solution_p - getSolution(y, x - 1)) - (source_p - getSource(y, x - 1)));
                }
                if (y < shape[0] - 1) {
                    append_error((solution_p - getSolution(y + 1, x)) - (source_p - getSource(y + 1, x)));
                }
                if (x < shape[1] - 1) {
                    append_error((solution_p - getSolution(y, x + 1)) - (source_p - getSource(y, x + 1)));
                }
            } else if (isBorder(y, x)) {
                dfdp = 4 * (getSolution(y, x) - getTarget(y, x));
            } else {
                float solution_p = getSolution(y, x);
                float target_p = getTarget(y, x);
                if (y > 0) {
                    append_error((solution_p - getSolution(y - 1, x)) - (target_p - getTarget(y - 1, x)));
                }
                if (x > 0) {
                    append_error((solution_p - getSolution(y, x - 1)) - (target_p - getTarget(y, x - 1)));
                }
                if (y < shape[0] - 1) {
                    append_error((solution_p - getSolution(y + 1, x)) - (target_p - getTarget(y + 1, x)));
                }
                if (x < shape[1] - 1) {
                    append_error((solution_p - getSolution(y, x + 1)) - (target_p - getTarget(y, x + 1)));
                }
            }

            #undef append_error

            getScratch(y, x) = dfdp;
            total_dfdp += fabs(dfdp);
        }
    }

    if (total_dfdp > 0) {
        for (y = 0; y < shape[0]; y++) {
            for (x = 0; x < shape[1]; x++) {
                getSolution(y, x) -= getScratch(y, x) * learning_rate / total_dfdp;
            }
        }
    }

    return total_error;
}


PyObject *
poisson_blend(PyObject *self, PyObject *args) {
    Quickdescent quickdescent;

    if (quickdescent.parseArgs(args))
        return NULL;

    if (quickdescent.initializeGuess())
        return NULL;

    if (quickdescent.blend())
        return NULL;

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
