#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>  
#include "symnmf.h"


/*Converts a PyObject representing a single data point into a C array of double values.*/
double* get_point(PyObject *pyPoint){
    int j;
    PyObject* entry;
    double* c_point;
    Py_ssize_t dimention;
    dimention = PyList_Size(pyPoint);
    c_point = (double*)calloc(dimention, sizeof(double));
    if (c_point == NULL) handleMemoryFail();
    dimention = PyList_Size(pyPoint);
    for(j = 0 ; j < dimention ; j++){
        entry = PyList_GetItem(pyPoint, j);
        c_point[j] = PyFloat_AsDouble(entry);
    }
    return c_point;
}

/*Converts a PyObject of lists (array of points) into a C array of double pointers,
 where each pointer represents a point.*/
double** convert_array(PyObject *pyarray){
    double** array;
    Py_ssize_t i,m;
    PyObject* point;
    double* c_point;
    m = PyList_Size(pyarray);

    array = (double**)calloc(m, sizeof(double));
    if (array == NULL) handleMemoryFail();
    for(i = 0; i < m; i++){
        point = PyList_GetItem(pyarray, i);
        c_point = get_point(point);
        array[i] = c_point;
    }
    return array;
}

/*Converts a 2D C array of doubles into a PyObject.*/
PyObject* C_to_py(double** array, int row, int col){
    PyObject *python_float, *PyList, *point;
    int i ,j ;
    PyList = PyList_New(row);
    for (i = 0; i < row ; i++){
        point = PyList_New(col);
        for (j = 0; j < col; j++){
            python_float = Py_BuildValue("d", array[i][j]);
            PyList_SetItem(point, j, python_float);
        }
        PyList_SetItem(PyList, i, point);
    }
    return PyList;
}

/*Python-exposed function that calculates the similarity matrix 
A for the input data using the function sym.*/
static PyObject* sym_c(PyObject *self, PyObject *args)
{
    PyObject *pyData,*pyA;
    double **data, **A;
    Py_ssize_t m;
    if(!PyArg_ParseTuple(args, "O", &pyData)) {
        return NULL; 
    }
        m = PyList_Size(pyData);
        n = m;
        d = PyList_Size(PyList_GetItem(pyData,0));
        data = convert_array(pyData);
        A = sym(data);
        pyA = C_to_py(A,m,m);
        freeArray(A,m);
        freeArray(data,m);
    return Py_BuildValue("O",pyA); 
}

/* Python-exposed function that calculates the Diagonal Degree Matrix 
D from the similarity matrix A.*/
static PyObject* ddg_c(PyObject *self, PyObject *args)
{
    PyObject *pyData, *pyD;
    double **data, **A , **D;
    Py_ssize_t m;
    if(!PyArg_ParseTuple(args, "O", &pyData)) {
        return NULL; 
    }
    m = PyList_Size(pyData);
    n = m;
    d = PyList_Size(PyList_GetItem(pyData,0));
    data = convert_array(pyData);
    A = sym(data);
    D = ddg(A);
    pyD = C_to_py(D,m,m);
    freeArray(A,m);
    freeArray(data,m);
    freeArray(D,m);
    return Py_BuildValue("O",pyD); 
}

/*Python-exposed function that calculates the normalized similarity matrix W,
 derived from A and D.*/
static PyObject* norm_c(PyObject *self, PyObject *args)
{
    PyObject *pyData, *pyW;
    Py_ssize_t m;
    double **data, **A , **D, **W;
    if(!PyArg_ParseTuple(args, "O", &pyData)) {
        return NULL; 
    }
    data = convert_array(pyData);
    m = PyList_Size(pyData);
    n = m;
    d = PyList_Size(PyList_GetItem(pyData,0));
    A = sym(data);
    D = ddg(A);
    W = norm(D,A);
    freeArray(A,m);
    freeArray(data,m);
    freeArray(D,m);
    pyW = C_to_py(W,m,m);
    freeArray(W,m);
    return Py_BuildValue("O",pyW); 
}

/*Python-exposed function that performs SymNMF. 
It computes the matrix H based on the provided matrices H and W using the function finalH.*/
static PyObject* symnmf_c(PyObject *self, PyObject *args)
{
    PyObject *pyH, *pyW, *pyAtH;
    double **H, **W, **res;
    int k;
    Py_ssize_t m;
    if(!PyArg_ParseTuple(args, "OOi", &pyH,&pyW,&k)) {
        return NULL; 
    }
        m = PyList_Size(pyW);
        n = m;
        H = convert_array(pyH);
        W = convert_array(pyW);
        res = finalH(H,W,k);        
        pyAtH = C_to_py(res,n,k);
        freeArray(H,n);
        freeArray(res,n);
        freeArray(W,n);
    return Py_BuildValue("O",pyAtH); 
}

static PyMethodDef methods[] = {
    {
        "sym_c", // name exposed to Python
        sym_c, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculates the similarity matrix" // documentation
    }, {
        "ddg_c", // name exposed to Python
        ddg_c, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculates the Diagonal Degree Matrix" // documentation
    }, {
        "norm_c", // name exposed to Python
        norm_c, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculate and output the normalized similarity matrix" // documentation
    }, {
        "symnmf_c", // name exposed to Python
        symnmf_c, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Perform full symNMF" // documentation
    },{
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",     // name of module exposed to Python
    NULL, // module documentation
    -1,
    methods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}