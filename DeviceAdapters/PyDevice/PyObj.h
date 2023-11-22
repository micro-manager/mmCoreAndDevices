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
@brief A smart pointer to a Python object

A PyObj wraps a PyObject* pointer in a smart pointer, taking care of automatic reference counting, GIL locking, and type conversions.
Access to the attributes of the Python object is provided through Get, Set, and HasAttribute.
Member functions can be called using CallMethod, and callable Python objects can be invoked using Call.

Because the common way of the Python API to report errors is to return a null pointer, the PyObj constructor is the main point where errors raised by the Python code are captured.

PyObj has static functions to initialize the Python interpreter and run scripts.
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
    explicit PyObj(const char* value) : PyObj(PyUnicode_FromString(value)) {}
    explicit PyObj(long value) : PyObj(PyLong_FromLong(value)) {}
    explicit PyObj(bool value) : PyObj(value ? Py_True : Py_False) {}

    // utility functions to convert to primitive types
    // note: if an error occurred during these functions, it will be logged in the g_errorMessage (also see CheckErrors) check for python 
    // note: the current thread must hold the GIL (see PyLock)
    template <class T> T as() const;
    template <> long as<long>() const {
        PyLock lock;
        auto retval = PyLong_AsLong(*this);
        if (retval == -1) // may be an error
            ReportError();
        return retval;
    }
    template <> bool as<bool>() const {
        return p_ == Py_True;
    }
    template <> double as<double>() const {
        PyLock lock;
        auto retval = PyFloat_AsDouble(*this);
        if (retval == -1.0) // may be an error
            ReportError();
        return retval;
    }
    template <> string as<string>() const {
        PyLock lock;
        auto as_str = PyObj(PyObject_Str(*this)); // convert any object to a to Python string by calling the str() function
        if (as_str) {
            auto as_bytes = PyObj(PyUnicode_AsUTF8String(as_str));
            if (as_bytes) {
                auto retval = PyBytes_AsString(as_bytes);
                if (retval) {
                    return retval;
                }
            }
        }
        ReportError();
        return string();
    }
    template <> PyObj as<PyObj>() const {
        return *this;
    }
    template <class T> void Set(const string& attribute, T value) {
        Set(attribute.c_str(), value);
    }
    template <class T> void Set(const char* attribute, T value) {
        PyLock lock;
        PyObject_SetAttrString(p_, attribute, PyObj(value));
        ReportError();
    }
    PyObj CallMember(const char* function) noexcept {
        PyLock lock;
        auto member = Borrow(PyObject_GetAttrString(p_, function));
        if (member)
            return PyObj(PyObject_CallNoArgs(member));
        else
            return member;
    }
    PyObj Call() const noexcept {
        PyLock lock;
        return PyObj(PyObject_CallNoArgs(p_));
    }
    PyObj Call(const PyObj& arg) const noexcept {
        PyLock lock;
        PyObject* arg0 = arg;
        return PyObj(PyObject_CallFunctionObjArgs(p_, arg0, NULL));
    }
    //    PyObj Call(const PyObj& arg) const noexcept {
//        PyLock lock;
//        return PyObj(PyObject_CallOneArg(p_, arg));
//    }
    /* for Python 3.9:
    template <class ...Args> PyObj Call(const PyObj& arg1, const Args&... args) const
    {
        PyLock lock;
        std::vector<PyObject*> arguments = { arg1, args... };
        return PyObj(PyObject_VectorCall(p_, arguments.data(), sizeof...(args), nullptr));
    }*/

    PyObj Get( const char* attribute) const noexcept {
        PyLock lock;
        return PyObj(PyObject_GetAttrString(p_, attribute));
    }
    PyObj Get(const string& attribute) const noexcept {
        return Get(attribute.c_str());
    }
    bool HasAttribute(const string& attribute) const noexcept {
        return HasAttribute(attribute.c_str());
    }
    bool HasAttribute(const char* attribute) const noexcept {
        PyLock lock;
        return PyObject_HasAttrString(p_, attribute);
    }

    /**
     * @brief Clear the reference (setting it to nullptr). If this is the last reference to the object, the object is destroyed.
    */
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

    /**
     * @brief Used to access the contained PyObject pointer
    */
    operator PyObject* () const {
        return p_ ? p_ : Py_None;
    }
    operator bool() const {
        return p_ != nullptr;
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
    PyObj operator * (const PyObj& other) const {
        PyLock lock;
        return PyObj(PyNumber_Multiply(p_, other));
    }
    PyObj operator / (const PyObj& other) const {
        PyLock lock;
        return PyObj(PyNumber_TrueDivide(p_, other));
    }
    /**
    * Takes a borrowed reference and wraps it in a PyObj smart pointer
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

   
    static bool InitializeInterpreter(const string& module_path) noexcept;
    static bool RunScript(const string& code, const string& file_name, const PyObj& locals) noexcept;

    /**
    * Checks if a Python error has occurred. If so, logs the error and resets the error state.
    * Note: Python error handling is very fragile, and it is essential to check for errors after every call to a Python API function. This usually happens automatically by converting the result to a PyObj (see PyObj constructor). Failure to reset the error state after a Python exception has occurred results in very strange behavior (unrelated fake errors popping up in unrelated parts of the program) or a complete crash of the program (this happens in some cases when throwing an exception without resetting the error state first).
    
      The errors are all concatenated as a single string. Also see PythonBridge::CheckError, since this is the place where the error list is copied to the MM CoreDebug log and reported to the end user.
    */
    static bool ReportError();
    static string g_errorMessage;
    static PyThreadState* g_threadState;
    static PyObj g_unit_ms;
    static PyObj g_unit_um;
    static PyObj g_traceback_to_string;
    static PyObj g_scan_devices;
    static PyObj g_main_module;
    static PyObj g_global_scope;
    static PyObj g_set_path;
};

