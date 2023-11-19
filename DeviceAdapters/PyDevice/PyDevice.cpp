///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The implementation of the Python camera. Adapted from the Democamera in the 
//                Micromanager repository.
//                
// AUTHOR:        Jeroen Doornbos
//                Ivo Vellekoop
// COPYRIGHT:     
// LICENSE:       ?
#include "pch.h"
#include "PyDevice.h"


std::map<string, CPyHub*> CPyHub::g_hubs;



int CPyDeviceBase::CheckError() noexcept {
    PyLock lock;
    PyObj::ReportError(); // check if any new errors happened
    if (!PyObj::g_errorMessage.empty()) {
        LogError(PyObj::g_errorMessage.c_str());
        PyObj::g_errorMessage.clear();
        return ERR_PYTHON_EXCEPTION;
    }
    else
        return DEVICE_OK;
}

CPyHub::CPyHub() : PyHubClass(g_adapterName) {
    SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not initialize Python interpreter, perhaps an incorrect path was specified?");
    SetErrorText(ERR_PYTHON_MULTIPLE_INTERPRETERS, "A different Python interpreter was already started. Due to limitations in the Python runtime, it is not possible to switch interpeters or virtual environments. Please fix the Python library path, or leave it blank.\n"
    "If you want to change the Python interpreter or virtual environment, you will have to restart Micro-Manager with a configuration of (none), and rebuild the configuration using the new interpreter.\n"
    "Alternatively, you can edit the .cfg file manually with a text editor.");
    CreateStringProperty(p_PythonScript, "", false, nullptr, true);
    CreateStringProperty(p_PythonHomePath, "", false, nullptr, true);
}


/**
 * Destroys all Python objects by releasing the references we currently have
 * 
 * @return 
*/
int CPyHub::Shutdown() {
    if (initialized_) {
        PyLock lock;
        devices_.clear();
        initialized_ = false;
        g_hubs.erase(id_);
    }
    return PyHubClass::Shutdown();
}

int CPyHub::DetectInstalledDevices() {
    ClearInstalledDevices();
    for (const auto& key_value : devices_) {
        auto mm_device = new CPyGenericDevice(key_value.first);
        AddInstalledDevice(mm_device);
    }
    return CheckError();
}

/**
 * @brief Initialize the Python interpreter, run the script, and convert the 'devices' dictionary into a c++ map
*/
int CPyHub::Initialize() {
    if (!initialized_) {
        char pythonExecutablePath[MM::MaxStrLength] = { 0 };
        char scriptPathString[MM::MaxStrLength] = { 0 };
        _check_(GetProperty(p_PythonHomePath, pythonExecutablePath));
        _check_(GetProperty(p_PythonScript, scriptPathString));

        fs::path scriptPath(scriptPathString);
        #ifdef _WIN32
        while (!FileExists(scriptPath)) {
            OPENFILENAMEA options = { 0 };
            char file_name[MAX_PATH] = { 0 };
            strncpy_s(file_name, scriptPath.generic_string().c_str(), MAX_PATH - 1);
            options.lStructSize = sizeof(OPENFILENAMEA);
            options.lpstrFilter = "Python scripts\0*.py\0\0";
            options.lpstrFile = file_name;
            options.lpstrTitle = "Select Python file that includes the `devices` dictionary";
            options.nMaxFile = MAX_PATH;

            if (!GetOpenFileNameA(&options))
                return ERR_PYTHON_NOT_FOUND;

            scriptPath = options.lpstrFile;
            _check_(SetProperty(p_PythonScript, scriptPath.generic_string().c_str()));
        }
        #endif

        PyObj::InitializeInterpreter(pythonExecutablePath);
        auto variables = PyObj::RunScript(scriptPathString);
        id_ = scriptPath.filename().generic_string();

        // read the 'devices' field, which must be a dictionary of label->device
        PyLock lock; // so that we can call bare Python API functions
        auto deviceDict = PyObj(PyDict_Items(PyObj::Borrow(PyDict_GetItemString(variables, "devices"))));
        auto device_count = PyList_Size(deviceDict);
        for (Py_ssize_t i = 0; i < device_count; i++) {
            auto key_value = PyObj::Borrow(PyList_GetItem(deviceDict, i));
            auto name = PyObj::Borrow(PyTuple_GetItem(key_value, 0)).as<string>();
            auto obj = PyObj::Borrow(PyTuple_GetItem(key_value, 1));
            auto type = obj.Get("_MM_dtype").as<string>();
            auto id = ComposeId(type, id_, name);
            obj.Set("_MM_id", id);
            devices_[id] = obj;
        }
        g_hubs[id_] = this;

        initialized_ = true;
    }
    return CheckError();
}





#define ATTRIBUTE_NAME 0    // internal name for Python (snake_case)
#define PROPERTY_NAME 1     // property name in MM (TitleCase)
#define TYPE 2
#define READ_ONLY 3
#define MIN 4
#define MAX 5
#define ENUMS 6

vector<PyAction*> CPyDeviceBase::EnumerateProperties() noexcept
{
    PyLock lock;
    auto propertyDescriptors = vector<PyAction*>();
    auto properties = object_.Get("_MM_properties");
    auto property_count = PyList_Size(properties);
    for (Py_ssize_t i = 0; i < property_count; i++) {
        PyAction* descriptor;

        auto pinfo = PyObj::Borrow(PyList_GetItem(properties, i));
        auto attrName = PyObj::Borrow(PyTuple_GetItem(pinfo, ATTRIBUTE_NAME)).as<string>();
        auto mmName = PyObj::Borrow(PyTuple_GetItem(pinfo, PROPERTY_NAME)).as<string>();
        auto type = PyObj::Borrow(PyTuple_GetItem(pinfo, TYPE)).as<string>();
        auto readonly = PyObj::Borrow(PyTuple_GetItem(pinfo, READ_ONLY)).as<bool>();

        if (type == "int")
            descriptor = new PyIntAction(this, attrName, mmName, readonly);
        else if (type == "float")
            descriptor = new PyFloatAction(this, attrName, mmName, readonly);
        else if (type == "time")
            descriptor = new PyQuantityAction(this, attrName, mmName, readonly, PyObj::g_unit_ms);
        else if (type == "length")
            descriptor = new PyQuantityAction(this, attrName, mmName, readonly, PyObj::g_unit_um);
        else if (type == "string")
            descriptor = new PyStringAction(this, attrName, mmName, readonly);
        else if (type == "bool")
            descriptor = new PyBoolAction(this, attrName, mmName, readonly);
        else if (type == "enum") {
            descriptor = new PyEnumAction(this, attrName, mmName, readonly);
            auto options = PyObj(PyDict_Items(PyObj::Borrow(PyTuple_GetItem(pinfo, ENUMS))));
            auto option_count = PyList_Size(options);
            for (Py_ssize_t j = 0; j < option_count; j++) {
                auto key_value = PyObj::Borrow(PyList_GetItem(options, j));
                descriptor->enum_keys.push_back(PyObj::Borrow(PyTuple_GetItem(key_value, 0)).as<string>());
                descriptor->enum_values.push_back(PyObj::Borrow(PyTuple_GetItem(key_value, 1)));
            }
        }
        else // other property type, treat as object
            descriptor = new PyObjectAction(this, attrName, mmName, readonly);
        
        if (descriptor->type == MM::Integer || descriptor->type == MM::Float) {
            auto lower = PyObj::Borrow(PyTuple_GetItem(pinfo, MIN));
            auto upper = PyObj::Borrow(PyTuple_GetItem(pinfo, MAX));
            if (lower != Py_None && upper != Py_None) {
                descriptor->min = lower.as<double>();
                descriptor->max = upper.as<double>();
                descriptor->has_limits = true;
            }
        }

        propertyDescriptors.push_back(descriptor);
    }
    return propertyDescriptors;
}


/**
* Locates a Python object by device id
*/
PyObj CPyHub::GetDevice(const string& device_id) noexcept {
    // split device id
    string deviceType;
    string hubId;
    string deviceName;
    CPyHub::SplitId(device_id, deviceType, hubId, deviceName);
    
    auto hub_idx = g_hubs.find(hubId);
    if (hub_idx == g_hubs.end())
        return PyObj(); // hub not found
    auto hub = hub_idx->second;

    auto device_idx = hub->devices_.find(device_id);
    if (device_idx == hub->devices_.end())
        return PyObj(); // device not found

    return device_idx->second;
}

/*
    * A device id is of the form DeviceType[name@hub], where :
    *  DeviceType is the device type : "Device", "Camera", etc.
    *  hub is the name of the script that was used to construct the device, including the .py suffix: "microscope.py"
    *  name is the key of the 'devices' dictionary that contains the object
    */
bool CPyHub::SplitId(const string& id, string& deviceType, string& hubId, string& deviceName) noexcept {
    auto colon1 = id.find('[');
    auto colon2 = id.find('@', colon1 + 1);
    if (colon1 != string::npos && colon2 != string::npos) {
        deviceType = id.substr(0, colon1);
        deviceName = id.substr(colon1 + 1, colon2 - colon1 - 1);
        hubId = id.substr(colon2 + 1, id.length() - colon2 - 2);
        return true;
    }
    else
        return false;
};

string CPyHub::ComposeId(const string& deviceType, const string& hubId, const string& deviceName) noexcept {
    return deviceType + "[" + deviceName + "@" + hubId + "]";
}
