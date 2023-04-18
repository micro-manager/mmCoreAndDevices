#pragma once
#include <string>
#include <functional>
#include <filesystem>
#include <limits>
#include "MMDeviceConstants.h"
#include "DeviceBase.h"

namespace fs = std::filesystem;
using std::string;
using std::function;
using std::numeric_limits;


// the following lines are a workaround for the problem 'cannot open file python39_d.lib'. This occurs because Python tries
// to link to the debug version of the library, even when that is not installed (and not really needed in our case).
// as a workaround, we trick the python.h include to think we are always building a Release build.
#ifdef _DEBUG
#undef _DEBUG
#define _HAD_DEBUG
#endif

// see https://numpy.org/doc/stable/reference/c-api/array.html#c.import_array
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PyDevice_ARRAY_API
#ifndef IMPORT_ARRAY_HERE
#define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>

// restore _DEBUG macro
#ifdef _HAD_DEBUG
#define _DEBUG
#endif




#define ERR_PYTHON_NOT_FOUND 101
#define ERR_PYTHON_PATH_CONFLICT 102
#define ERR_PYTHON_SCRIPT_NOT_FOUND 103
#define ERR_PYTHON_CLASS_NOT_FOUND 104
#define ERR_PYTHON_EXCEPTION 105
#define ERR_PYTHON_NO_INFO 106
#define ERR_PYTHON_MISSING_PROPERTY 107
#define _check_(expression) { auto result = expression; if (result != DEVICE_OK) return result; }


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
* todo: implement move constructor
*/
class PyObj {
    PyObject* _p;
public:
    PyObj() : _p(nullptr) {
    }
    PyObj(PyObj&& other) noexcept : _p(other._p)  {
        other._p = nullptr;
    }

    /**
    * Takes a new reference and wraps it into a PyObj smart pointer
    * This does not increase the reference count of the object
    * The reference count is decreased when the PyObj smart pointer is destroyed (or goes out of scope).
    * 
    * Throws an exception when obj == NULL, because this is the common way of the Python API to report errors
    */
    explicit PyObj(PyObject* obj) : _p(obj) {
        if (!obj)
            throw PythonException();
    }
    void Clear() {
        if (_p) {
            PyLock lock;
            Py_DECREF(_p);
            _p = nullptr;
        }
    }
    PyObj(const PyObj& other) : _p(other) {
        if (_p) {
            PyLock lock;
            Py_INCREF(_p);
        }
    }
    ~PyObj() {
        Clear();
    }
    operator PyObject* () const { 
        return _p;
    }
    PyObject* get() const {
        return _p;
    }
    PyObj& operator = (PyObj&& other) noexcept {
        Clear();
        _p = other._p;
        other._p = nullptr;
        return *this;
    }
    

    /**
    * Takes a borrowed reference and wraps it into a PyObj smart pointer
    * This increases the reference count of the object.
    * The reference count is decreased when the PyObj smart pointer is destroyed (or goes out of scope).
    * 
    * Throws an exception when obj == NULL, because this is the common way of the Python API to report errors
    */
    static PyObj FromBorrowed(PyObject* obj) {
        if (obj) {
            PyLock lock;
            Py_INCREF(obj);
        }
        return PyObj(obj);
    }
    class PythonException : public std::exception {
    };
    PyObj& operator = (const PyObj& other) {
        if (_p || other._p) {
            PyLock lock;
            Py_XDECREF(_p);
            _p = other;
            Py_XINCREF(_p);
        }
        return *this;
    }
private:
    void incref();
    void decref();
};


class PythonBridge
{
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
    static constexpr const char* p_PythonDeviceClass = "Device class";
    static unsigned int g_ActiveDeviceCount;
    static fs::path g_PythonHome;
    static PyObj g_Module;
     
    PyObj _object;
    PyObj _options;
    PyObj _intPropertyType;
    PyObj _floatPropertyType;
    PyObj _stringPropertyType;
    bool initialized_;
    PyThreadState* threadState_;
    const string _name;
    const function<void(const char*)> _errorCallback;
    
public:
    int InitializeInterpreter(const char* pythonHome) noexcept;
    int Destruct() noexcept;
   
    int SetProperty(const char* name, long value) noexcept;
    int SetProperty(const char* name, double value) noexcept;
    int SetProperty(const char* name, const string& value) noexcept;
    int GetProperty(const char* name, long& value) const noexcept;
    int GetProperty(const char* name, double& value) const noexcept;
    int GetProperty(const char* name, string& value) const noexcept;
    PyObj GetProperty(const char* name) const; //@todo: make noexcept

    static bool PythonActive() noexcept {
        return g_ActiveDeviceCount > 0;
    }

    PythonBridge(const function<void(const char*)>& errorCallback) : _errorCallback(errorCallback), initialized_(false), threadState_(nullptr) {
    }

    PyObj Call(const PyObj& callable) {
        PyLock lock;
        return PyObj(PyObject_CallNoArgs(callable));
    }

    /** Sets up init-only properties on the MM device
    * Properties for locating the Python libraries, the Python script, and for the name of the device class are added. No Python calls are made
      @todo: leave python home blank as default (meaning 'auto locate'). Add verification handler when home path set
      @todo: add verification handler when script path set
    */
    template <class T> void Construct(CDeviceBase<T, PythonBridge>* device) {
        device->CreateStringProperty(p_PythonHome, PythonBridge::FindPython().generic_string().c_str(), false, nullptr, true);
        device->CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        device->CreateStringProperty(p_PythonDeviceClass, "Device", false, nullptr, true);
    }

    template <class T> int Initialize(CDeviceBase<T, PythonBridge>* device) {
        if (initialized_)
            return DEVICE_OK;

        char pythonHome[MM::MaxStrLength] = { 0 };
        char pythonScript[MM::MaxStrLength] = { 0 };
        char pythonDeviceClass[MM::MaxStrLength] = { 0 };
        _check_(device->GetProperty(p_PythonHome, pythonHome));
        _check_(device->GetProperty(p_PythonScript, pythonScript));
        _check_(device->GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(InitializeInterpreter(pythonHome));
        PyLock lock;
        _check_(ConstructPythonObject(pythonScript, pythonDeviceClass));
        g_ActiveDeviceCount++;
        initialized_ = true;

        _check_(CreateProperties(device));
        return DEVICE_OK;
    }

    int OnString(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        if (eAct == MM::AfterSet)
        {
            string value;
            pProp->Get(value);
            return SetProperty(pProp->GetName().c_str(), value);
        }
        return DEVICE_OK;
    }

    int OnFloat(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
        if (eAct == MM::AfterSet)
        {
            double value;
            pProp->Get(value);
            return SetProperty(pProp->GetName().c_str(), value);
        }
        return DEVICE_OK;
    }
    int OnInteger(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
        if (eAct == MM::AfterSet)
        {
            long value;
            pProp->Get(value);
            return SetProperty(pProp->GetName().c_str(), value);
        }
        return DEVICE_OK;
    }
    int PythonError() const;
private:
    static fs::path FindPython() noexcept;
    static bool HasPython(const fs::path& path) noexcept;
    int ConstructPythonObject(const char* pythonScript, const char* pythonClass) noexcept;
    static long GetInt(PyObject* object, const char* string); //todo: make noexcept
    static PyObj GetAttr(PyObject* object, const char* string);
    static double GetFloat(PyObject* object, const char* string);
    static string GetString(PyObject* object, const char* string);
    static string PyUTF8(PyObject* obj);

    template <class T> int CreateProperties(CDeviceBase<T, PythonBridge>* device) noexcept {
        using Action = typename CDeviceBase<T, PythonBridge>::CPropertyAction;

        try {
            auto property_count = PyList_Size(_options);
            for (Py_ssize_t i = 0; i < property_count; i++) {
                auto key_value = PyList_GetItem(_options, i); // note: borrowed reference, don't ref count (what a mess...)
                auto name = PyUTF8(PyTuple_GetItem(key_value, 0));
                if (name.empty())
                    continue;

                // construct int/float/string property
                auto property = PyTuple_GetItem(key_value, 1);
                if (PyObject_IsInstance(property, _intPropertyType)) {
                    device->CreateIntegerProperty(name.c_str(), GetInt(_object, name.c_str()), false, new Action(this, &PythonBridge::OnInteger));
                }
                else if (PyObject_IsInstance(property, _floatPropertyType)) {
                    device->CreateFloatProperty(name.c_str(), GetFloat(_object, name.c_str()), false, new Action(this, &PythonBridge::OnFloat));
                }
                else if (PyObject_IsInstance(property, _stringPropertyType)) {
                    device->CreateStringProperty(name.c_str(), GetString(_object, name.c_str()).c_str(), false, new Action(this, &PythonBridge::OnString));
                }
                else
                    continue;

                // Set limits. Only supported by MM if both upper and lower limit are present.
                auto lower = PyObject_HasAttrString(property, "min") ? GetFloat(property, "min") : -std::numeric_limits<double>().infinity();
                auto upper = PyObject_HasAttrString(property, "max") ? GetFloat(property, "max") : std::numeric_limits<double>().infinity();
                if (isfinite(lower) && isfinite(upper))
                    device->SetPropertyLimits(name.c_str(), lower, upper);
            }
        }
        catch (PyObj::PythonException) {
            return PythonError();
        }
        return DEVICE_OK;
    }
};

