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
    operator bool () const {
        return p_ != nullptr;
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
    PyObj& operator = (const PyObj& other) {
        if (p_ || other.p_) {
            PyLock lock;
            Py_XDECREF(p_);
            p_ = other;
            Py_XINCREF(p_);
        }
        return *this;
    }
    class PythonException : public std::exception {
    };
};


class PythonBridge
{
    struct Link {
        PyObj object;
        string attribute;
        string value;
    };
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
    static constexpr const char* p_PythonDeviceClass = "Device class";
    static bool g_initializedInterpreter;
    static PyThreadState* g_threadState;
    static fs::path g_PythonHome;
    static std::unordered_map<string, PyObject*> g_Devices;
    static std::vector<Link> g_MissingLinks;
    
    PyObj module_;
    PyObj object_;
    PyObj options_;
    PyObj intPropertyType_;
    PyObj floatPropertyType_;
    PyObj stringPropertyType_;
    PyObj objectPropertyType_;
    bool initialized_;
    string label_;
    const function<void(const char*)> errorCallback_;
    
public:
    int InitializeInterpreter(const char* pythonHome) noexcept;
    int Destruct() noexcept;
   
    int SetProperty(const char* name, long value) noexcept;
    int SetProperty(const char* name, double value) noexcept;
    int SetProperty(const char* name, const string& value) noexcept;
    int SetProperty(const char* name, PyObject* value) noexcept;
    template <class T> int GetProperty(const char* name, T& value) const noexcept {
        return Get(object_, name, value);
    }

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
        char label[MM::MaxStrLength] = { 0 };
        _check_(device->GetProperty(p_PythonHome, pythonHome));
        _check_(device->GetProperty(p_PythonScript, pythonScript));
        _check_(device->GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(InitializeInterpreter(pythonHome));
        PyLock lock;
        _check_(ConstructPythonObject(pythonScript, pythonDeviceClass));
        initialized_ = true;

        _check_(CreateProperties(device));

        device->GetLabel(label); // Note: it seems that SetLabel is only called once, and before device initialization, so we can safely read it here and consider it constant.
        label_ = label;
        Register(); // register device in device map, and link to existing python objects if needed
        return DEVICE_OK;
    }

    int PythonError() const;
private:
    PythonBridge(const PythonBridge& other) = delete; // no copy constructor
    static fs::path FindPython() noexcept;
    static bool HasPython(const fs::path& path) noexcept;
    int ConstructPythonObject(const char* pythonScript, const char* pythonClass) noexcept;
    int Get(PyObject* object, const char* name, PyObj& value) const noexcept;
    int Get(PyObject* object, const char* name, long& value) const noexcept;
    int Get(PyObject* object, const char* name, double& value) const noexcept;
    int Get(PyObject* object, const char* name, std::string& value) const noexcept;
    static string PyUTF8(PyObject* obj);
    void Register() const;

    template <class T> int CreateProperties(CDeviceBase<T, PythonBridge>* device) noexcept {
        using Action = typename CDeviceBase<T, PythonBridge>::CPropertyAction;

        // Traverse the options_ dictionary and find all properties of types we recognize.
        // Convert these to MM int/float/string properties that will show up in the property browser.
        // Note: we can be sure options_ is a PyDict. Therefore, we can be sure that the PyList and PyTuple functions succeed.
        // we do need to check if the keys and values are of the correct type
        auto property_count = PyList_Size(options_);
        for (Py_ssize_t i = 0; i < property_count; i++) {
            auto key_value = PyList_GetItem(options_, i); // note: borrowed reference, don't ref count (what a mess...)
            auto name = PyUTF8(PyTuple_GetItem(key_value, 0));
            if (name.empty())
                continue;   // key was not a string
            auto property = PyTuple_GetItem(key_value, 1);

            if (PyObject_IsInstance(property, intPropertyType_)) {
                long value;
                _check_(Get(object_, name.c_str(), value));
                _check_(device->CreateIntegerProperty(name.c_str(), value, false, new Action(this, &PythonBridge::OnProperty<long>)));
            }
            else if (PyObject_IsInstance(property, floatPropertyType_)) {
                double value;
                _check_(Get(object_, name.c_str(), value));
                _check_(device->CreateFloatProperty(name.c_str(), value, false, new Action(this, &PythonBridge::OnProperty<double>)));
            }
            else if (PyObject_IsInstance(property, stringPropertyType_)) {
                string value;
                _check_(Get(object_, name.c_str(), value));
                _check_(device->CreateStringProperty(name.c_str(), value.c_str(), false, new Action(this, &PythonBridge::OnProperty<string>)));
            } if (PyObject_IsInstance(property, objectPropertyType_)) {
                _check_(device->CreateStringProperty(name.c_str(), "", false, new Action(this, &PythonBridge::OnObjectProperty)));
            }
            else
                continue;

            // Set limits. Only supported by MM if both upper and lower limit are present.
            // The min/max attributes are always present, we only need to check if they don't hold 'None'
            PyObj lower, upper;
            _check_(Get(property, "min", lower));
            _check_(Get(property, "max", upper));
            if (lower != Py_None && upper != Py_None) {
                double lower_val, upper_val;
                _check_(Get(property, "min", lower_val));
                _check_(Get(property, "max", upper_val));
                device->SetPropertyLimits(name.c_str(), lower_val, upper_val);
            }
        }
        return DEVICE_OK;
    }

    /**
     * Callback that is called when a property value is read or written
     * We respond to property writes by relaying the value to the Python object.
     * In this implementation, we don't perform any action when the value is read. We just use the cached value in MM.
     * Note that this is not correct if the property is changed by Python.
     * @return MM result code
    */
    template <class T> int OnProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
        if (eAct == MM::AfterSet)
        {
            T value = {};
            pProp->Get(value);
            return SetProperty(pProp->GetName().c_str(), value);
        }
        return DEVICE_OK;
    }

    /**
     * Callback that is called when an object property value is read or written
     * This property holds a string corresponding to the MM label of a PyDevice object.
     * When the property is set, look up the corresponding Python object and store a reference to that object in the Python property.
     * Note: Unfortunately, we cannot use MM's built in object map because we cannot cast a MM::Device to a CPyDevice object because of the Curiously Recurring Template pattern used by CGenericBase. Therefore, we have to keep a list of devices ourselves.
    */
    int OnObjectProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
        if (eAct == MM::AfterSet)
        {
            string label;
            pProp->Get(label);
            auto device = g_Devices.find(label); // look up device by name
            if (device != g_Devices.end()) {
                return SetProperty(pProp->GetName().c_str(), device->second);
            }
            else { // label not found. This could be because the object is not constructed yet
                g_MissingLinks.push_back({ object_, pProp->GetName(), label });
            }
        }
        return DEVICE_OK;
    }
};

