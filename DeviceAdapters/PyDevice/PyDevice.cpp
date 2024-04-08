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

CPyHub* CPyHub::g_the_hub = nullptr;

CPyHub::CPyHub() : PyHubClass(g_adapterName) {
    SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the Python script.");
    SetErrorText(ERR_PYTHON_NO_DEVICE_DICT, "Script did not generate a global `device` variable holding a dictionary.");
    SetErrorText(ERR_PYTHON_ONLY_ONE_HUB_ALLOWED, "Only one PyHub device may be active at a time. To combine multiple Python devices, write a script that combines them in a single `devices` dictionary.");
    
    CreateStringProperty(p_PythonScriptPath, "", false, nullptr, true);
    CreateStringProperty(p_PythonModulePath, "(auto)", false, nullptr, true);
    id_ = "PyHub";
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
        g_the_hub = nullptr;
        PyObj::DeinitializeInterpreter();
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
 * @brief Loads the Python script specified in ScriptPath
 * If the script is not found, a dialog box file browser is shown so that the user can select a file (Windows only).
 * @return Script text string read from the file, or an empty string if the user cancelled the load. 
*/
string CPyHub::LoadScript() noexcept {
    char scriptPathString[MM::MaxStrLength] = { 0 };
    if (GetProperty(p_PythonScriptPath, scriptPathString) != DEVICE_OK)
        return string();

    script_path_ = scriptPathString;
#ifdef _WIN32
    if (!FileExists(script_path_)) { // file not found, let the user select one
        OPENFILENAMEW options = { 0 };
        wchar_t file_name[MAX_PATH] = { 0 };
        wcsncpy(file_name, script_path_.generic_wstring().c_str(), MAX_PATH - 1);
        options.lStructSize = sizeof(OPENFILENAMEW);
        options.lpstrFilter = L"Python scripts\0*.py\0\0";
        options.lpstrFile = file_name;
        options.lpstrTitle = L"Select Python file that includes the `devices` dictionary";
        options.nMaxFile = MAX_PATH;

        if (GetOpenFileName(&options)) {
            script_path_ = options.lpstrFile;
            if (SetProperty(p_PythonScriptPath, script_path_.generic_u8string().c_str()) != DEVICE_OK)
                return string();
        }
    }
#endif

    // load the python script from disk
    auto stream = std::ifstream(script_path_);
    if (!stream)
        return string(); // file not found

    auto code = std::string();
    char buffer[1024];
    while (stream.read(buffer, sizeof(buffer)))
        code.append(buffer, 0, stream.gcount());
    
    code.append(buffer, 0, stream.gcount());
    return code;
}

/**
  @brief Initializes additional paths where Python looks for modules.
  The module search path always includes the default paths as set up by Py_Initialize.
  This includes the site-packages folder of the Python installation that is currently used.

  If ModulePath is set to "(auto)" (the default), the following search paths are added at the start of the module search path:
  - The directory where the script is located
  - If that directory, or any of the parent directories, holds a `venv` folder, use the site-packages in that virtual environment

  If ModulePath is not set to "(auto)", the paths set in the ModulePath (separated by ';') are added at the start of the module search path.

  Note: these paths are set *before* calling the bootstrap script. That script should be able to locate the numpy and astropy packages.
  Note: if the hub is de-initialized and initialized again, the paths are reset.
*/
string CPyHub::ComputeModulePath() noexcept {
    char modulePathString[MM::MaxStrLength] = { 0 };
    if (GetProperty(p_PythonModulePath, modulePathString) != DEVICE_OK)
        return string();

    auto path = string(modulePathString);
    if (path == "(auto)") {
        path = script_path_.parent_path().generic_u8string(); // always include the folder of the current script

        // see if the script is 'in' a virtual environment. If so, use that environment
        struct stat info;
        fs::path dir = script_path_;
        for (int depth = 0; depth < 10 && dir.has_relative_path(); depth++) {
            dir = dir.parent_path();
            stat((dir / "venv").generic_u8string().c_str(), &info);
            if (info.st_mode & S_IFDIR) {
                path += ';';
                path += dir.generic_u8string();
                path += "/venv/Lib/site-packages";
                break;
            }
        }
    }
    return path;
}

/**
 * @brief Initialize the Python interpreter, run the script, and convert the 'devices' dictionary into a c++ map
*/
int CPyHub::Initialize() {
    if (!initialized_) {
        if (g_the_hub)
            return ERR_PYTHON_ONLY_ONE_HUB_ALLOWED;
        //
        // Read path to the Python script, and optional path to the Python home directory (virtual environment)
        // If the Python script is not found, a dialog box is shown so that the user can select a file.
        //

        auto script = LoadScript();
        if (script.empty())
            return ERR_PYTHON_SCRIPT_NOT_FOUND;
       

        auto modulePath = ComputeModulePath();

        this->LogMessage("Initializing the Python runtime. The Python runtime (especially Anaconda) may crash if Python is not installed correctly."
            "If so, verify thatthe HOMEPATH environment is set to the correct value, or remove it."
            "Also, make sure that the desired Python installation is the first that is listed in the PATH environment variable.\n", true);

        if (!PyObj::InitializeInterpreter())
            return CheckError(); // initializing the interpreter failed, abort initialization and report the error

        PyLock lock;
        // execute the Python script, and read the 'devices' field,
        auto deviceDict = PyObj::g_load_devices.Call(PyObj(modulePath), PyObj(script_path_.parent_path().generic_u8string()), PyObj(script_path_.stem().generic_u8string()));
        if (!deviceDict)
            return CheckError();
        
        // process device list and add metadata
        auto deviceList = PyObj(PyDict_Items(deviceDict));
        auto device_count = PyList_Size(deviceList); // todo: move to PyObj? to assert lock?
        for (Py_ssize_t i = 0; i < device_count; i++) {
            auto key_value = deviceList.GetListItem(i);
            auto name = key_value.GetTupleItem(0).as<string>();
            auto obj = key_value.GetTupleItem(1);
            auto type = obj.Get("device_type").as<string>();
            auto id = ComposeId(type, name);
            obj.Set("_MM_id", id);
            devices_[id] = obj;
        }

        initialized_ = true;
        g_the_hub = this;
    }
    return CheckError();
}

tuple<vector<PyAction*>, PyObj> EnumerateProperties(const PyObj& deviceInfo, const ErrorCallback& callback) noexcept
{
    PyLock lock;

    // Loop over all properties in the PyDevice Python object, and convert the properties to Action objects
    // These objects can be used as callbacks for the MM property system, and used directly to get/set property values.
    //
    auto propertyDescriptors = vector<PyAction*>();
    auto properties = deviceInfo.Get("properties");
    auto property_count = PyList_Size(properties);
    for (Py_ssize_t i = 0; i < property_count; i++) {
        PyAction* descriptor;
        auto pinfo = properties.GetListItem(i);
        auto mmName = pinfo.Get("mm_name").as<string>();
        auto getter = pinfo.Get("get");
        auto setter = pinfo.Get("set");
        auto type = pinfo.Get("data_type").as<string>();

        if (type == "int")
            descriptor = new PyIntAction(getter, setter, mmName, callback);
        else if (type == "float")
            descriptor = new PyFloatAction(getter, setter, mmName, callback);
        else if (type == "str")
            descriptor = new PyStringAction(getter, setter, mmName, callback);
        else if (type == "bool")
            descriptor = new PyBoolAction(getter, setter, mmName, callback);
        else if (type == "enum") {
            descriptor = new PyEnumAction(getter, setter, mmName, callback);
            auto options = PyObj(PyDict_Items(pinfo.Get("options")));
            auto option_count = PyList_Size(options);
            for (Py_ssize_t j = 0; j < option_count; j++) {
                auto key_value = options.GetListItem(j);
                descriptor->enum_keys.push_back(key_value.GetTupleItem(0).as<string>());
                descriptor->enum_values.push_back(key_value.GetTupleItem(1));
            }
        }
        else // other property type, skip
            continue;
        
        if (descriptor->type == MM::Integer || descriptor->type == MM::Float) {
            auto lower = pinfo.Get("min");
            auto upper = pinfo.Get("max");
            if (lower != Py_None && upper != Py_None) {
                descriptor->min = lower.as<double>();
                descriptor->max = upper.as<double>();
                descriptor->has_limits = true;
            }
        }

        propertyDescriptors.push_back(descriptor);
    }

    auto methods = deviceInfo.Get("methods");
    
    return tuple(propertyDescriptors, methods);
}


/**
* Locates a Python object by device id
*/
PyObj CPyHub::GetDeviceInfo(const string& device_id) noexcept {
    // split device id
    string deviceType;
    string deviceName;
    CPyHub::SplitId(device_id, deviceType, deviceName);
    if (!g_the_hub)
        return PyObj(); // no hub initialized

    auto device_idx = g_the_hub->devices_.find(device_id);
    if (device_idx == g_the_hub->devices_.end())
        return PyObj(); // device not found

    return device_idx->second;
}

/*
    * A device id is of the form DeviceType:name, where :
    *  DeviceType is the device type : "Device", "Camera", etc.
    *  name is the key of the 'devices' dictionary that contains the object
    */
bool CPyHub::SplitId(const string& id, string& deviceType, string& deviceName) noexcept {
    auto colon1 = id.find(':');
    if (colon1 != string::npos) {
        deviceType = id.substr(0, colon1);
        deviceName = id.substr(colon1 + 1);
        return true;
    }
    else
        return false;
};

string CPyHub::ComposeId(const string& deviceType, const string& deviceName) noexcept {
    return deviceType + ":" + deviceName;
}
