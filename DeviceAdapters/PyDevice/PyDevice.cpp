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
    SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the Python script.");
    SetErrorText(ERR_PYTHON_NO_DEVICE_DICT, "Script did not generate a global `device` variable holding a dictionary.");
    
    CreateStringProperty(p_PythonScriptPath, "", false, nullptr, true);
    CreateStringProperty(p_PythonModulePath, "", false, nullptr, true);
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
        //
        // Read path to the Python script, and optional path to the Python home directory (virtual environment)
        // If the Python script is not found, a dialog box is shown so that the user can select a file.
        //
        char modulePathString[MM::MaxStrLength] = { 0 };
        char scriptPathString[MM::MaxStrLength] = { 0 };
        _check_(GetProperty(p_PythonModulePath, modulePathString));
        _check_(GetProperty(p_PythonScriptPath, scriptPathString));

        fs::path scriptPath(scriptPathString);
        #ifdef _WIN32
        if (!FileExists(scriptPath)) {
            OPENFILENAMEA options = { 0 };
            char file_name[MAX_PATH] = { 0 };
            strncpy_s(file_name, scriptPath.generic_string().c_str(), MAX_PATH - 1);
            options.lStructSize = sizeof(OPENFILENAMEA);
            options.lpstrFilter = "Python scripts\0*.py\0\0";
            options.lpstrFile = file_name;
            options.lpstrTitle = "Select Python file that includes the `devices` dictionary";
            options.nMaxFile = MAX_PATH;

            if (GetOpenFileNameA(&options)) {
               scriptPath = options.lpstrFile;
                _check_(SetProperty(p_PythonScriptPath, scriptPath.generic_string().c_str()));
            }            
        }
        #endif

        // load the python script from disk
        auto stream = std::ifstream(scriptPath);
        if (!stream)
            return ERR_PYTHON_SCRIPT_NOT_FOUND; 

        auto code = std::string();
        char buffer[1024];
        while (stream.read(buffer, sizeof(buffer))) {
            code.append(buffer, 0, stream.gcount());
        }
        code.append(buffer, 0, stream.gcount());

        id_ = scriptPath.filename().generic_string();
        
        this->LogMessage("Initializing the Python runtime. The Python runtime (especially Anaconda) may crash if Python is not installed correctly."
            "If so, verify thatthe HOMEPATH environment is set to the correct value, or remove it."
            "Also, make sure that the desired Python installation is the first that is listed in the PATH environment variable.\n", true);

        if (!PyObj::InitializeInterpreter(modulePathString))
            return CheckError(); // initializing the interpreter failed, abort initialization and report the error

        // execute the Python script, and read the 'devices' field,
        // which must be a dictionary of {label: device}
        PyLock lock; // lock, so that we can call bare Python API functions
        auto locals = PyObj(PyDict_New());
        PyObj::g_add_to_path.Call(PyObj(scriptPath.parent_path().generic_u8string().c_str()));
        if (!PyObj::RunScript(code.c_str(), id_.c_str(), locals))
            return CheckError();

        auto deviceDict = PyObj::Borrow(PyDict_GetItemString(locals, "devices"));
        if (!deviceDict)
            return ERR_PYTHON_NO_DEVICE_DICT;
        
        // process device list and add metadata
        auto deviceList = PyObj(PyDict_Items(PyObj::g_scan_devices.Call(deviceDict)));
        if (!deviceList)
            return CheckError();

        auto device_count = PyList_Size(deviceList);
        for (Py_ssize_t i = 0; i < device_count; i++) {
            auto key_value = PyObj::Borrow(PyList_GetItem(deviceList, i));
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
