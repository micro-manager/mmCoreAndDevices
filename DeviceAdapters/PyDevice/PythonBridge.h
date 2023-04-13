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
};

struct PythonProperty {
    string name;
    MM::PropertyType type;
    double lower_limit = -numeric_limits<double>::infinity();
    double upper_limit = numeric_limits<double>::infinity();

    bool HasLimits() const {
        return isfinite(lower_limit) || isfinite(upper_limit);
    }
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
    MM::Core* _core;
    function<void(const char*)> _errorCallback;
    
public:
    PythonBridge();
    int Initialize(const char* pythonHome, const char* pythonScript, const char* pythonClass);
    int Destruct();
   
    int SetProperty(const string& name, long value);
    int SetProperty(const string& name, double value);
    int SetProperty(const string& name, const string& value);

    std::vector<PythonProperty> EnumerateProperties();
    static bool PythonActive() {
        return g_ActiveDeviceCount > 0;
    }
    static fs::path FindPython();

    template <class T> void Construct(CDeviceBase<T, PythonBridge>* device) {
        // Adds properties for locating the Python libraries, the Python script, and the name of the device class
        device->CreateStringProperty(p_PythonHome, PythonBridge::FindPython().generic_string().c_str(), false, nullptr, true);
        device->CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        device->CreateStringProperty(p_PythonDeviceClass, "Device", false, nullptr, true);
    }

    template <class T> int Initialize(CDeviceBase<T, PythonBridge>* device, MM::Core* core) {
        using Action = typename CDeviceBase<T, PythonBridge>::CPropertyAction;

        // Set up error callback
        _core = core;
        if (_core)
            _errorCallback = [device, core](const char* message) { core->LogMessage(device, message, false); };

        char pythonHome[MM::MaxStrLength] = { 0 };
        char pythonScript[MM::MaxStrLength] = { 0 };
        char pythonDeviceClass[MM::MaxStrLength] = { 0 };
        _check_(device->GetProperty(p_PythonHome, pythonHome));
        _check_(device->GetProperty(p_PythonScript, pythonScript));
        _check_(device->GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(Initialize(pythonHome, pythonScript, pythonDeviceClass));

        for (const auto& option : EnumerateProperties()) {
            switch (option.type) {
            case MM::String:
                device->CreateStringProperty(option.name.c_str(), "", false, new Action(this, &PythonBridge::OnString));
                break;
            case MM::Integer:
                device->CreateIntegerProperty(option.name.c_str(), 0, false, new Action(this, &PythonBridge::OnInteger));
                break;
            case MM::Float:
                device->CreateFloatProperty(option.name.c_str(), 0.0, false, new Action(this, &PythonBridge::OnFloat));
                break;
            }
            if (option.HasLimits())
                device->SetPropertyLimits(option.name.c_str(), option.lower_limit, option.upper_limit);
        }
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
    int ConstructInternal(const char* pythonScript, const char* pythonClass);
    static string PyUTF8(PyObject* obj);
};

