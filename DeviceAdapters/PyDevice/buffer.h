#pragma once
#include "stable.h"

struct Py_buffer {
    void* buf;
    PyObject* obj;        /* owned reference */
    Py_ssize_t len;
    Py_ssize_t itemsize;  /* This is Py_ssize_t so it can be
                             pointed to by strides in simple case.*/
    int readonly;
    int ndim;
    char* format;
    Py_ssize_t* shape;
    Py_ssize_t* strides;
    Py_ssize_t* suboffsets;
    void* internal;
};

extern "C" {
    int PyObject_CheckBuffer(PyObject* obj);
    int PyObject_GetBuffer(PyObject* exporter, Py_buffer* view, int flags);
    void PyBuffer_Release(Py_buffer* view);
    Py_ssize_t PyBuffer_SizeFromFormat(const char* format);
    int PyBuffer_IsContiguous(const Py_buffer* view, char order);
    void* PyBuffer_GetPointer(const Py_buffer* view, const Py_ssize_t* indices);
    int PyBuffer_FromContiguous(const Py_buffer* view, const void* buf, Py_ssize_t len, char fort);
    int PyBuffer_ToContiguous(void* buf, const Py_buffer* src, Py_ssize_t len, char order);
    int PyObject_CopyData(PyObject* dest, PyObject* src);
    void PyBuffer_FillContiguousStrides(int ndims, Py_ssize_t* shape, Py_ssize_t* strides, int itemsize, char order);
    int PyBuffer_FillInfo(Py_buffer* view, PyObject* exporter, void* buf, Py_ssize_t len, int readonly, int flags);
}


typedef int (*getbufferproc)(PyObject*, Py_buffer*, int);
typedef void (*releasebufferproc)(PyObject*, Py_buffer*);

typedef PyObject* (*vectorcallfunc)(PyObject* callable, PyObject* const* args,
    size_t nargsf, PyObject* kwnames);

/* Maximum number of dimensions */
#define PyBUF_MAX_NDIM 64

/* Flags for getting buffers */
#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001
/*  we used to include an E, backwards compatible alias  */
#define PyBUF_WRITEABLE PyBUF_WRITABLE
#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)

#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)

#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)

#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)


#define PyBUF_READ  0x100
#define PyBUF_WRITE 0x200
