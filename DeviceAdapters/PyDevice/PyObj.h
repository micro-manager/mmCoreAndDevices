#pragma once
#include "pch.h"

/**
 * Helper class to automatically lock and unlock the global interpreter lock (GIL)
 * This is needed because Python is single threaded (!) while MM is not.
 * Note that the GIL should be locked for any Python call, including Py_INCREF and Py_DECREF
*/
class PyLock {
    PyGILState_STATE gstate_;
public:
    PyLock() {
        gstate_ = PyGILState_Ensure();
    }
    ~PyLock() {
        PyGILState_Release(gstate_);
    }
};

/**
* Smart pointer object to automate reference counting of PyObject* pointers
*
* Because the common way of the Python API to report errors is to return a null pointer, the PyObj constructor is the main point where errors raised by the Python code are captured.
*/
class PyObj {
    PyObject* p_;
public:
    PyObj() : p_(nullptr) {
    }
    PyObj(PyObj&& other) noexcept : p_(other.p_) {
        other.p_ = nullptr;
    }

    explicit PyObj(PyObject* obj);
    PyObj(const PyObj& other) : p_(other ? other : nullptr) {
        if (p_) {
            PyLock lock;
            Py_INCREF(p_);
        }
    }

    // utility functions to construct new python object from primitive values
    // note: the current thread must hold the GIL (see PyLock)
    explicit PyObj(double value) : PyObj(PyFloat_FromDouble(value)) {}
    explicit PyObj(const string& value) : PyObj(PyUnicode_FromString(value.c_str())) {}
    explicit PyObj(long value) : PyObj(PyLong_FromLong(value)) {}

    // utility functions to convert to primitive types
    // note: if an error occurred during these functions, it will be logged in the g_errorMessage (also see CheckErrors) check for python 
    // note: the current thread must hold the GIL (see PyLock)
    template <class T> T as() const;
    template <> long as<long>() const {
        auto retval = PyLong_AsLong(*this);
        if (retval == -1) // may be an error
            ReportError();
        return retval;
    }
    template <> double as<double>() const {
        auto retval = PyFloat_AsDouble(*this);
        if (retval == -1.0) // may be an error
            ReportError();
        return retval;
    }
    template <> string as<string>() const {
        auto retval = PyUnicode_AsUTF8(*this);
        if (!retval) { // error
            ReportError();
            return string();
        }
        return retval;
    }
    template <> PyObj as<PyObj>() const {
        return *this;
    }
    
    void Clear() {
        if (p_) {
            PyLock lock;
            Py_DECREF(p_);
            p_ = nullptr;
        }
    }
    ~PyObj() {
        Clear();
    }
    operator PyObject* () const {
        return p_ ? p_ : Py_None;
    }
    operator bool() const {
        return p_ != nullptr;
    }
    PyObject* get() const {
        return p_ == nullptr ? Py_None : p_;
    }
    PyObj& operator = (PyObj&& other) noexcept {
        Clear();
        p_ = other.p_;
        other.p_ = nullptr;
        return *this;
    }
    PyObj& operator = (const PyObj& other) {
        if (p_ || other.p_) {
            PyLock lock;
            Py_XDECREF(p_);
            p_ = other;
            Py_XINCREF(p_);
        }
        return *this;
    }

    /**
    * Takes a borrowed reference and wraps it into a PyObj smart pointer
    * This increases the reference count of the object.
    * The reference count is decreased when the PyObj smart pointer is destroyed (or goes out of scope).
    */
    static PyObj Borrow(PyObject* obj) {
        if (obj) {
            PyLock lock;
            Py_INCREF(obj);
        }
        return PyObj(obj);
    }

    /**
    * Checks if a Python error has occurred. If so, logs the error and resets the error state.
    * Note: Python error handling is very fragile, and it is essential to check for errors after every call to a Python API function. This usually happens automatically by converting the result to a PyObj (see PyObj constructor). Failure to reset the error state after a Python exception has occurred results in very strange behavior (unrelated fake errors popping up in unrelated parts of the program) or a complete crash of the program (this happens in some cases when throwing an exception without resetting the error state first).
    
      The errors are all concatenated as a single string. Also see PythonBridge::CheckError, since this is the place where the error list is copied to the MM CoreDebug log and reported to the end user.
    */
    static void ReportError();
    static string g_errorMessage;
};
