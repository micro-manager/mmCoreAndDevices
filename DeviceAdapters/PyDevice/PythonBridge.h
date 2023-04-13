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


class PyObj {
    PyObject* _p;
public:
    PyObj() : _p(nullptr) {
    }
    explicit PyObj(PyObject* obj) : _p(obj) {
        if (!obj)
            throw new NullPointerException();
        Py_XINCREF(_p);
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
    PyObj& operator = (const PyObj& other) {
        Py_XDECREF(_p);
        _p = other;
        Py_XINCREF(_p);
        return *this;
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
    string _name;
    function<void(const char*)> _errorCallback;
    
public:
    PythonBridge();
    int InitializeInterpreter(const char* pythonHome);
    int Destruct();
   
    int SetProperty(const string& name, long value);
    int SetProperty(const string& name, double value);
    int SetProperty(const string& name, const string& value);

    static bool PythonActive() {
        return g_ActiveDeviceCount > 0;
    }
    static fs::path FindPython();

    PythonBridge(const function<void(const char*)>& errorCallback) : _errorCallback(errorCallback) {
    }


    template <class T> void Construct(CDeviceBase<T, PythonBridge>* device) {
        // Adds properties for locating the Python libraries, the Python script, and the name of the device class
        device->CreateStringProperty(p_PythonHome, PythonBridge::FindPython().generic_string().c_str(), false, nullptr, true);
        device->CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        device->CreateStringProperty(p_PythonDeviceClass, "Device", false, nullptr, true);
    }

    template <class T> int Initialize(CDeviceBase<T, PythonBridge>* device) {
        char pythonHome[MM::MaxStrLength] = { 0 };
        char pythonScript[MM::MaxStrLength] = { 0 };
        char pythonDeviceClass[MM::MaxStrLength] = { 0 };
        _check_(device->GetProperty(p_PythonHome, pythonHome));
        _check_(device->GetProperty(p_PythonScript, pythonScript));
        _check_(device->GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(InitializeInterpreter(pythonHome));
        _check_(ConstructPythonObject(pythonScript, pythonDeviceClass));
        g_ActiveDeviceCount++;
        _check_(CreateProperties(device));
        return DEVICE_OK;
    }

    int OnString(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        if (eAct == MM::AfterSet)
        {
            string value;
            pProp->Get(value);
            return SetProperty(pProp->GetName(), value);
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
            return SetProperty(pProp->GetName(), value);
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
            return SetProperty(pProp->GetName(), value);
        }
        return DEVICE_OK;
    }


    //static string DefaultPluginPath();
private:
    static bool HasPython(const fs::path& path);
    int PythonError();
    int ConstructPythonObject(const char* pythonScript, const char* pythonClass);
    static PyObj GetAttr(PyObject* object, const char* string);
    static long GetInt(PyObject* object, const char* string);
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
                    auto defaultValue = GetInt(property, "default");
                    device->CreateIntegerProperty(name.c_str(), defaultValue, false, new Action(this, &PythonBridge::OnInteger));
                }
                else if (PyObject_IsInstance(property, _floatPropertyType)) {
                    auto defaultValue = GetFloat(property, "default");
                    device->CreateFloatProperty(name.c_str(), defaultValue, false, new Action(this, &PythonBridge::OnFloat));
                }
                else if (PyObject_IsInstance(property, _stringPropertyType)) {
                    auto defaultValue = GetString(property, "default");
                    device->CreateStringProperty(name.c_str(), defaultValue.c_str(), false, new Action(this, &PythonBridge::OnString));
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

