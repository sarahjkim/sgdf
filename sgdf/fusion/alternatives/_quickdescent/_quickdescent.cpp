#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"


class Quickdescent {
        PyArrayObject *arr_source, *arr_mask, *arr_tinyt, *arr_solution, *arr_scratch,
                      *arr_errorlog;
        size_t shape[2];

    public:
        Quickdescent() {};
        ~Quickdescent() {
            Py_XDECREF(arr_source);
            Py_XDECREF(arr_mask);
            Py_XDECREF(arr_tinyt);
            Py_XDECREF(arr_solution);
            Py_XDECREF(arr_scratch);
            Py_XDECREF(arr_errorlog);
        };
        int parseArgs(PyObject *args);
        PyObject *initializeGuess();
        PyObject *blend(PyObject *args);
    private:
        float averageBorderValue(PyArrayObject *im);
        float descend(float learning_rate, float epsilon, int max_iterations);
        inline bool isBorder(npy_intp y, npy_intp x);
        inline float getSource(const npy_intp y, const npy_intp x);
        inline bool getMask(const npy_intp y, const npy_intp x);
        inline float getTarget(const npy_intp y, const npy_intp x);
        inline float &getSolution(const npy_intp y, const npy_intp x);
        inline float &getScratch(const npy_intp y, const npy_intp x);
        inline float &getErrorlog(const npy_intp n);
};


typedef struct {
    PyObject_HEAD
    Quickdescent *quickdescent;
} QuickdescentContext_object;


static void
QuickdescentContext_dealloc(QuickdescentContext_object *self) {
    if (self->quickdescent) {
        self->quickdescent->~Quickdescent();
        PyMem_Free(self->quickdescent);
    }
    self->ob_type->tp_free((PyObject *)self);
}


static PyObject *
QuickdescentContext_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    QuickdescentContext_object *self = (QuickdescentContext_object *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->quickdescent = (Quickdescent *) PyMem_Malloc(sizeof(Quickdescent));
        if (self->quickdescent == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        memset(self->quickdescent, 0, sizeof(Quickdescent));
        new (self->quickdescent) Quickdescent();
    }
    return (PyObject *)self;
}


static int
QuickdescentContext_init(QuickdescentContext_object *self, PyObject *args, PyObject *kwargs) {
    if (self->quickdescent == NULL) {
        self->quickdescent = (Quickdescent *) PyMem_Malloc(sizeof(Quickdescent));
        if (self->quickdescent == NULL) {
            return -1;
        }
        memset(self->quickdescent, 0, sizeof(Quickdescent));
        new (self->quickdescent) Quickdescent();
    }

    int success = self->quickdescent->parseArgs(args);

    /* Converting return code conventions. */
    if (success == 0) {
        return 0;
    } else {
        return -1;
    }
}


static PyObject *
QuickdescentContext_initializeGuess(QuickdescentContext_object *self) {
    return self->quickdescent->initializeGuess();
}


static PyObject *
QuickdescentContext_blend(QuickdescentContext_object *self, PyObject *args) {
    return self->quickdescent->blend(args);
}


static PyMethodDef QuickdescentContext_methods[] = {
    {"initializeGuess", (PyCFunction)QuickdescentContext_initializeGuess, METH_NOARGS,
        "Initializes the solution, based on a heuristic."},
    {"blend", (PyCFunction)QuickdescentContext_blend, METH_VARARGS,
        "Perform gradient descent on the solution."},
    {NULL}
};


static PyTypeObject QuickdescentContext_type {
    PyObject_HEAD_INIT(NULL)
    0,                                       /* ob_size */
    "_quickdescent.QuickdescentContext",     /* tp_name */
    sizeof(QuickdescentContext_object),      /* tp_basicsize */
    0,                                       /* tp_itemsize */
    (destructor)QuickdescentContext_dealloc, /* tp_dealloc */
    0,                                       /* tp_print */
    0,                                       /* tp_getattr */
    0,                                       /* tp_setattr */
    0,                                       /* tp_compare */
    0,                                       /* tp_repr */
    0,                                       /* tp_as_number */
    0,                                       /* tp_as_sequence */
    0,                                       /* tp_as_mapping */
    0,                                       /* tp_hash  */
    0,                                       /* tp_call */
    0,                                       /* tp_str */
    0,                                       /* tp_getattro */
    0,                                       /* tp_setattro */
    0,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "A Quickdescent instance",               /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    QuickdescentContext_methods,             /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    (initproc)QuickdescentContext_init,      /* tp_init */
    0,                                       /* tp_alloc */
    QuickdescentContext_new,                 /* tp_new */
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
    if (!PyArg_ParseTuple(args, "OOOO!O!O!", &arg_source, &arg_mask, &arg_tinyt,
                          &PyArray_Type, &arg_solution, &PyArray_Type, &arg_scratch,
                          &PyArray_Type, &arg_errorlog)) {
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

    arr_errorlog = (PyArrayObject *) PyArray_FROM_OTF(arg_errorlog, NPY_FLOAT, NPY_ARRAY_INOUT_ARRAY);
    if (arr_errorlog == NULL) {
        return 1;
    }
    if (PyArray_NDIM(arr_errorlog) != 1) {
        PyErr_SetString(PyExc_ValueError, "Error log should be a 1 dimensional array.");
        return 1;
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
PyObject *
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
    Py_INCREF(arr_solution);
    return (PyObject *)arr_solution;
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


PyObject *
Quickdescent::blend(PyObject *args) {
    float epsilon;
    int max_iterations;
    if (!PyArg_ParseTuple(args, "fi", &epsilon, &max_iterations)) {
        return NULL;
    }

    int errorlog_size = PyArray_DIMS(arr_errorlog)[0];
    if (max_iterations == 0) {
        max_iterations = errorlog_size;
    } else if (max_iterations > errorlog_size) {
        PyErr_Format(PyExc_ValueError, "max_iterations is %d, but errorlog only has space for %u",
                     max_iterations, errorlog_size);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    float error, previous_error, delta_error;
    float learning_rate;
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        learning_rate = 100.0 * max_iterations / (max_iterations + 500.0 * iteration);
        error = descend(learning_rate, epsilon, max_iterations);
        getErrorlog(iteration) = error;
        if (iteration > 0) {
            delta_error = error - previous_error;
            if (fabs(delta_error) <= epsilon)
                break;
        }
        previous_error = error;
    }
    Py_END_ALLOW_THREADS

    Py_INCREF(arr_solution);
    return (PyObject *)arr_solution;
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
Quickdescent::descend(float learning_rate, float epsilon, int max_iterations) {
    npy_intp y, x;
    double total_error = 0, total_dfdp = 0;
    const size_t y_max = shape[0] - 1;
    const size_t x_max = shape[1] - 1;
    const size_t row_size = shape[1];
    const float *ptr_source = (float *)PyArray_DATA(arr_source);
    const bool *ptr_mask = (bool *)PyArray_DATA(arr_mask);
    const float *ptr_tinyt = (float *)PyArray_DATA(arr_tinyt);
    float *ptr_solution = (float *)PyArray_DATA(arr_solution);
    float *ptr_scratch = (float *)PyArray_DATA(arr_scratch);

    /* Here are optimized versions of our accessor methods that we use in this tight loop.
     * The PyArray_FROM_OTF will guarantee that our arrays are C-style contiguous, so we can
     *     address them directly as arrays.
     * This optimization provides roughly 2x to 3x improvement in performance.  */
    #define getSource(y, x) (ptr_source[(y) * row_size + (x)])
    #define getMask(y, x) (ptr_mask[(y) * row_size + (x)])
    #define getTarget(y, x) (ptr_tinyt[(y) * row_size + (x)])
    #define getSolution(y, x) (ptr_solution[(y) * row_size + (x)])
    #define getScratch(y, x) (ptr_scratch[(y) * row_size + (x)])
    #define isBorderAssumingNotMask(y, x) (((y) > 0 && getMask((y) - 1, (x))) || \
                                           ((y) < y_max && getMask((y) + 1, (x))) || \
                                           ((x) > 0 && getMask((y), (x) - 1)) || \
                                           ((x) < x_max && getMask((y), (x) + 1)))

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
            } else if (isBorderAssumingNotMask(y, x)) {
                dfdp = 4 * (getSolution(y, x) - getTarget(y, x));
            }

            #undef append_error

            getScratch(y, x) = dfdp;
            total_dfdp += fabs(dfdp);
        }
    }

    if (total_dfdp > 0) {
        for (y = 0; y < shape[0]; y++) {
            for (x = 0; x < shape[1]; x++) {
                if (getMask(y, x) || isBorderAssumingNotMask(y, x)) {
                    getSolution(y, x) -= getScratch(y, x) * learning_rate / total_dfdp;
                }
            }
        }
    }

    #undef getSource
    #undef getMask
    #undef getTarget
    #undef getSolution
    #undef getScratch
    #undef isBorderAssumingNotMask

    return total_error;
}


static PyMethodDef functions[] = {{NULL}};


PyMODINIT_FUNC
init_quickdescent(void) {
    PyObject *module;
    QuickdescentContext_type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&QuickdescentContext_type) < 0) {
        return;
    }

    module = Py_InitModule3("_quickdescent", functions,
                            "Native extensions for the quickdescent algorithm");
    Py_INCREF(&QuickdescentContext_type);
    PyModule_AddObject(module, "QuickdescentContext", (PyObject *)&QuickdescentContext_type);

    /* Sets up the Numpy API */
    import_array();
}
