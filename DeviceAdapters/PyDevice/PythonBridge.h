#pragma once
#include "pch.h"
#include "PyObj.h"


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
    static std::unordered_map<string, PyObj> g_Devices;
    static std::vector<Link> g_MissingLinks;
    
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
    PythonBridge(const function<void(const char*)>& errorCallback) : errorCallback_(errorCallback), initialized_(false) {
    }

    int Call(const PyObj& callable, PyObj& retval) const noexcept;

    template <class T> int GetProperty(const char* name, T& value) const noexcept {
        return Get(object_, name, value);
    }
    
    template <class T> int SetProperty(const char* name, T value) const noexcept {
        PyLock lock;
        PyObject_SetAttrString(object_, name, PyObj(value));
        return CheckError();
    }
    /**
     * Checks if a Python error has occurred since the last call to CheckError
     * @return DEVICE_OK or ERR_PYTHON_EXCEPTION
    */
    int CheckError() const {
        PyObj::ReportError(); // check if any new errors happened
        if (!PyObj::g_errorMessage.empty()) {
            errorCallback_(PyObj::g_errorMessage.c_str());
            PyObj::g_errorMessage.clear();
            return ERR_PYTHON_EXCEPTION;
        }
        else
            return DEVICE_OK;
    }

    /** Sets up init-only properties on the MM device
    * Properties for locating the Python libraries, the Python script, and for the name of the device class are added. No Python calls are made
      @todo: leave python home blank as default (meaning 'auto locate'). Add verification handler when home path set
      @todo: add verification handler when script path set
    */
    template <class T> void Construct(CDeviceBase<T, PythonBridge>* device, const char* defaultClassName) {
        device->CreateStringProperty(p_PythonHome, PythonBridge::FindPython().generic_string().c_str(), false, nullptr, true);
        device->CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        device->CreateStringProperty(p_PythonDeviceClass, defaultClassName, false, nullptr, true); // remove 'Py' prefix
    }
    int Destruct() noexcept;   

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
        _check_(CreateProperties(device));

        device->GetLabel(label); // Note: it seems that SetLabel is only called once, and before device initialization, so we can safely read it here and consider it constant.
        Register(); // register device in device map, and link to existing python objects if needed
        initialized_ = true;
        label_ = label;
        return DEVICE_OK;
    }
    static string PyUTF8(PyObject* obj);
private:
    PythonBridge(const PythonBridge& other) = delete; // no copy constructor
    static fs::path FindPython() noexcept;
    static bool HasPython(const fs::path& path) noexcept;
    int ConstructPythonObject(const char* pythonScript, const char* pythonClass) noexcept;
    int Get(PyObject* object, const char* name, PyObj& value) const noexcept;
    int Get(PyObject* object, const char* name, long& value) const noexcept;
    int Get(PyObject* object, const char* name, double& value) const noexcept;
    int Get(PyObject* object, const char* name, std::string& value) const noexcept;
    int InitializeInterpreter(const char* pythonHome) noexcept;
    static void UpdateLastError();
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
            } else if (PyObject_IsInstance(property, objectPropertyType_)) {
                _check_(device->CreateStringProperty(name.c_str(), "", false, new Action(this, &PythonBridge::OnObjectProperty)));
                continue; // skip setting limits or enum
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

            // For enum-type objects (may be string, int or float), notify MM about the allowed values
            // The allowed_values attribute is always present, we only need to check if they don't hold 'None'
            PyObj allowed_values;
            _check_(Get(property, "allowed_values", allowed_values));
            if (allowed_values != Py_None) {
                std::vector<std::string> allowed_value_strings;
                auto value_count = PyList_Size(allowed_values);
                for (Py_ssize_t j = 0; j < value_count; j++) {
                    auto value = PyList_GetItem(allowed_values, j); // borrowed reference, don't ref count
                    allowed_value_strings.push_back(PyUTF8(PyObj(PyObject_Str(value))));
                }
                device->SetAllowedValues(name.c_str(), allowed_value_strings);
            }
        }
        return CheckError();
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

