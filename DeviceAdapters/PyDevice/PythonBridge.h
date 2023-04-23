#pragma once
#include "pch.h"
#include <string>
#include <functional>
#include <filesystem>
#include <MMDeviceConstants.h>
#include <DeviceBase.h>

namespace fs = std::filesystem;
using std::string;
using std::function;

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
    PyObject* p_;
public:
    PyObj() : p_(nullptr) {
    }
    PyObj(PyObj&& other) noexcept : p_(other.p_)  {
        other.p_ = nullptr;
    }

    /**
    * Takes a new reference and wraps it into a PyObj smart pointer
    * This does not increase the reference count of the object
    * The reference count is decreased when the PyObj smart pointer is destroyed (or goes out of scope).
    * 
    * Throws an exception when obj == NULL, because this is the common way of the Python API to report errors
    */
    explicit PyObj(PyObject* obj) : p_(obj) {
        if (!obj)
            throw PythonException();
    }
    void Clear() {
        if (p_) {
            PyLock lock;
            Py_DECREF(p_);
            p_ = nullptr;
        }
    }
    PyObj(const PyObj& other) : p_(other) {
        if (p_) {
            PyLock lock;
            Py_INCREF(p_);
        }
    }
    ~PyObj() {
        Clear();
    }
    operator PyObject* () const { 
        return p_;
    }
    PyObject* get() const {
        return p_;
    }
    PyObj& operator = (PyObj&& other) noexcept {
        Clear();
        p_ = other.p_;
        other.p_ = nullptr;
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
        if (p_ || other.p_) {
            PyLock lock;
            Py_XDECREF(p_);
            p_ = other;
            Py_XINCREF(p_);
        }
        return *this;
    }
};


class PythonBridge
{
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
    static constexpr const char* p_PythonDeviceClass = "Device class";
    static bool g_initializedInterpreter;
    static PyThreadState* g_threadState;
    static fs::path g_PythonHome;
    
    PyObj module_;
    PyObj object_;
    PyObj options_;
    PyObj intPropertyType_;
    PyObj floatPropertyType_;
    PyObj stringPropertyType_;
    bool initialized_;
    const string name_;
    const function<void(const char*)> errorCallback_;
    
public:
    int InitializeInterpreter(const char* pythonHome) noexcept;
    int Destruct() noexcept;
   
    int SetProperty(const char* name, long value) noexcept;
    int SetProperty(const char* name, double value) noexcept;
    int SetProperty(const char* name, const string& value) noexcept;
    int GetProperty(const char* name, long& value) const noexcept;
    int GetProperty(const char* name, double& value) const noexcept;
    int GetProperty(const char* name, string& value) const noexcept;
    int GetProperty(const char* name, PyObj& value) const noexcept;

    PythonBridge(const function<void(const char*)>& errorCallback) : errorCallback_(errorCallback), initialized_(false) {
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
    int GetAttr(PyObject* object, const char* name, PyObj& value) const noexcept;
    int GetInt(PyObject* object, const char* name, long& value) const noexcept;
    int GetFloat(PyObject* object, const char* name, double& value) const noexcept;
    int GetString(PyObject* object, const char* name, std::string& value) const noexcept;
    static string PyUTF8(PyObject* obj);

    template <class T> int CreateProperties(CDeviceBase<T, PythonBridge>* device) noexcept {
        using Action = typename CDeviceBase<T, PythonBridge>::CPropertyAction;

        auto property_count = PyList_Size(options_);
        for (Py_ssize_t i = 0; i < property_count; i++) {
            auto key_value = PyList_GetItem(options_, i); // note: borrowed reference, don't ref count (what a mess...)
            auto name = PyUTF8(PyTuple_GetItem(key_value, 0));
            if (name.empty())
                continue;

            // construct int/float/string property
            auto property = PyTuple_GetItem(key_value, 1);
            if (!property)
                continue;

            if (PyObject_IsInstance(property, intPropertyType_)) {
                long value;
                _check_(GetInt(object_, name.c_str(), value));
                _check_(device->CreateIntegerProperty(name.c_str(), value, false, new Action(this, &PythonBridge::OnInteger)));
            }
            else if (PyObject_IsInstance(property, floatPropertyType_)) {
                double value;
                _check_(GetFloat(object_, name.c_str(), value));
                _check_(device->CreateFloatProperty(name.c_str(), value, false, new Action(this, &PythonBridge::OnFloat)));
            }
            else if (PyObject_IsInstance(property, stringPropertyType_)) {
                string value;
                _check_(GetString(object_, name.c_str(), value));
                _check_(device->CreateStringProperty(name.c_str(), value.c_str(), false, new Action(this, &PythonBridge::OnString)));
            }
            else
                continue;

            // Set limits. Only supported by MM if both upper and lower limit are present.
            // The min/max attributes are always present, we only need to check if they don't hold 'None'
            PyObj lower, upper;
            _check_(GetAttr(property, "min", lower));
            _check_(GetAttr(property, "max", upper));
            if (lower != Py_None && upper != Py_None) {
                double lower_val, upper_val;
                _check_(GetFloat(property, "min", lower_val));
                _check_(GetFloat(property, "max", upper_val));
                device->SetPropertyLimits(name.c_str(), lower_val, upper_val);
            }
        }
        return DEVICE_OK;
    }
};

