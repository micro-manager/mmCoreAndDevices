#include <string>
#include <functional>
#include <filesystem>
#include <vector>
#include <limits>
#include "MMDeviceConstants.h"
#include "DeviceBase.h"

namespace fs = std::filesystem;

#define ERR_PYTHON_NOT_FOUND 101
#define ERR_PYTHON_PATH_CONFLICT 102
#define ERR_PYTHON_SCRIPT_NOT_FOUND 103
#define ERR_PYTHON_CLASS_NOT_FOUND 104
#define ERR_PYTHON_EXCEPTION 105
#define ERR_PYTHON_NO_INFO 106
#define ERR_PYTHON_MISSING_PROPERTY 107

#pragma once
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#define _DEBUG
#else
#include <Python.h>
#endif
using std::string;
using std::function;
using std::numeric_limits;
#define _check_(expression) { auto result = expression; if (result != DEVICE_OK) return result; }


/**
* Smart pointer object to automate reference counting of PyObject* pointers
* todo: implement move constructor
*/
class PyObj {
    PyObject* _p;
public:
    PyObj() : _p(nullptr) {
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
            throw NullPointerException();
    }
    void Clear() {
        Py_XDECREF(_p);
        _p = nullptr;
    }
    PyObj(const PyObj& other) : _p(other) {
        Py_XINCREF(_p);
    }
    ~PyObj() {
        Py_XDECREF(_p);
    }
    operator PyObject* () const { 
        return _p;
    }
    PyObject* get() const {
        return _p;
    }
    PyObj& operator = (const PyObj& other) {
        Py_XDECREF(_p);
        _p = other;
        Py_XINCREF(_p);
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
        Py_XINCREF(obj);
        return PyObj(obj);
    }
    class NullPointerException : public std::exception {
    };
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
    const string _name;
    const function<void(const char*)> _errorCallback;
    
public:
    PythonBridge();
    int InitializeInterpreter(const char* pythonHome);
    int Destruct();
   
    int SetProperty(const char* name, long value);
    int SetProperty(const char* name, double value);
    int SetProperty(const char* name, const string& value);
    int GetProperty(const char* name, long& value) const;
    int GetProperty(const char* name, double& value) const;
    int GetProperty(const char* name, string& value) const;
    PyObj GetProperty(const char* name) const;

    static bool PythonActive() {
        return g_ActiveDeviceCount > 0;
    }
    static fs::path FindPython();

    PythonBridge(const function<void(const char*)>& errorCallback) : _errorCallback(errorCallback), initialized_(false) {
    }

    PyObj CallMethod(const PyObj& boundMethod) {
        auto result = PyObject_CallNoArgs(boundMethod);
        if (!result) {
            PythonError();
        }
        return PyObj(result);
    }

    template <class T> void Construct(CDeviceBase<T, PythonBridge>* device) {
        // Adds properties for locating the Python libraries, the Python script, and the name of the device class
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
        g_ActiveDeviceCount++;
        initialized_ = true;
        
        _check_(ConstructPythonObject(pythonScript, pythonDeviceClass));
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
    static bool HasPython(const fs::path& path);
    int ConstructPythonObject(const char* pythonScript, const char* pythonClass);
    static long GetInt(PyObject* object, const char* string);
    static PyObj GetAttr(PyObject* object, const char* string);
    static double GetFloat(PyObject* object, const char* string);
    static string GetString(PyObject* object, const char* string);
    static string PyUTF8(PyObject* obj);
    template <class T> int CreateProperties(CDeviceBase<T, PythonBridge>* device) {
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

                // set limits
                auto lower = PyObject_HasAttrString(property, "min") ? GetFloat(property, "min") : -std::numeric_limits<double>().infinity();
                auto upper = PyObject_HasAttrString(property, "max") ? GetFloat(property, "max") : std::numeric_limits<double>().infinity();
                if (isfinite(lower) || isfinite(upper))
                    device->SetPropertyLimits(name.c_str(), lower, upper);
            }
        }
        catch (PyObj::NullPointerException) {
            return PythonError();
        }
        return DEVICE_OK;
    }
};

