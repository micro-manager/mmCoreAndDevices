
#include "pch.h"
#include "PythonBridge.h"
#include <numpy/arrayobject.h>


bool PythonBridge::g_initializedInterpreter = false;

/** Stores the thread state when releasing the global interpreter lock for the first time. 
* Currently this value is never used, but is may be used to properly shut down the Python interpreter.*/
PyThreadState* PythonBridge::g_threadState = nullptr; 

/** Path of the currently active Python interpreter
* All devices must have the same Python interpreter path. This variable is set when the first device is initialized, and used to check if subsequent devices use the same path.
*/
fs::path PythonBridge::g_PythonHome;

/** Map of all PyDevice objects. Used to find an object by its label
* Maintained by ConstructObject and Destruct. Updated by SetLabel
*/
std::unordered_map<string, PyObj> PythonBridge::g_Devices;
std::vector<PythonBridge::Link> PythonBridge::g_MissingLinks;


/**
 * Initialize the Python interpreter
 * If a non-empty pythonHome path is specified, the python install from that path is used. All devices must use the same value for this path, or leave the value empty. If different values are provided, a ERR_PYTHON_PATH_CONFLICT is returned.
 * If a Python iterpreter is already running, this function only checks if the path is consistent and does nothing else.
 * Note: currently the Python interperter is never de-initialize (only when the process closes).
 * @param pythonHome location of the Python (virtual) installation to use. The folder must contain the python dll. Leave at "" for default (note: crashes anaconda 3.9!!). Must not be NULL
 * @todo test virtual environments
 * @return MM error/success code
*/
int PythonBridge::InitializeInterpreter(const char* pythonHome) noexcept
{
    // Check if Python is already initialized
    auto homePath = fs::path(pythonHome);
    if (g_initializedInterpreter) {
        if (homePath == g_PythonHome)
            return DEVICE_OK;
        else
            return ERR_PYTHON_PATH_CONFLICT;
    }

    // Initialize Python configuration (new style)
    // The old style initialization (using Py_Initialize) does not have a way to report errors. In particular,
    // if the Python installation cannot be found, the program just terminates!
    // The new style initialization returns an error, that can then be shown to the user instead of crashing micro manager.
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    if (!homePath.empty()) 
        PyConfig_SetString(&config, &config.home, homePath.c_str());
    auto status = Py_InitializeFromConfig(&config);

    //PyConfig_Read(&config); // for debugging
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status))
        return ERR_PYTHON_NOT_FOUND;


    g_threadState = PyEval_SaveThread(); // allow multi threading
    g_PythonHome = homePath;
    g_initializedInterpreter = true;

    PyLock lock;
    _import_array(); // initialize numpy. We don't use import_array (without _) because it hides any error message that may occur.
    return CheckError();
}

// Load python script
// This is done by constructing a python loader script and executing it.

/**
 * Loads the Python script and creates a device object
 * @param pythonScript path of the .py script file
 * @param pythonClass name of the Python class to create an instance of
 * @return MM return code
*/
int PythonBridge::ConstructPythonObject(const char* pythonScript, const char* pythonClass) noexcept {
    //assert(PyLock::IsLocked())
    auto scriptPath = fs::absolute(fs::path(pythonScript));
    auto bootstrap = std::stringstream();
    bootstrap <<
        "import numpy as np\n"
        "import sys\n"
        "sys.path.append('" << scriptPath.parent_path().generic_string() << "')\n"
        "code = open('" << scriptPath.generic_string() << "')\n"
        "exec(code.read())\n"
        "code.close()\n"
        "device = " << pythonClass << "()";

    auto scope = PyObj(PyDict_New()); // create a scope to execute the scripts in
    auto bootstrap_result = PyObj(PyRun_String(bootstrap.str().c_str(), Py_file_input, scope, scope));
    object_ = PyObj::Borrow(PyDict_GetItemString(scope, "device"));
    intPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "int_property"));
    floatPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "float_property"));
    stringPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "string_property"));
    objectPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "object_property"));
    return CheckError(); // check python errors one more time just to be sure
}


/**
* Checks if a Python error has occurred since the last call to CheckError
* @return DEVICE_OK or ERR_PYTHON_EXCEPTION
*/

/**
* Checks if a Python error has occurred since the last call to CheckError
* @return DEVICE_OK or ERR_PYTHON_EXCEPTION
*/
int PythonBridge::CheckError() const {
    PyObj::ReportError(); // check if any new errors happened
    if (!PyObj::g_errorMessage.empty()) {
        errorCallback_(PyObj::g_errorMessage.c_str());
        PyObj::g_errorMessage.clear();
        return ERR_PYTHON_EXCEPTION;
    }
    else
        return DEVICE_OK;
}

/**
 * Destroys the Python object by removing all references to it
 * If there are other references
 * @return 
*/
int PythonBridge::Destruct() noexcept {
    PyLock lock;
    object_.Clear();
    intPropertyType_.Clear();
    floatPropertyType_.Clear();
    stringPropertyType_.Clear();
    objectPropertyType_.Clear();
    initialized_ = false;

    // remove device from map
    g_Devices.erase(label_);
    return DEVICE_OK;
}

void PythonBridge::Register() const {
    PyLock lock; // also acts as lock for g_Device and g_MissingLinks

    // store device in global list
    g_Devices[label_] = object_;

    // check if there are unresolved links (c++ sucks)
    g_MissingLinks.erase(std::remove_if(g_MissingLinks.begin(), g_MissingLinks.end(), [this](PythonBridge::Link& l) {
        if (l.value == label_) {
            PyObject_SetAttrString(l.object, l.attribute.c_str(), object_);
            return true;
        }
        else
            return false;
    }), g_MissingLinks.end());
}

/**
* Callback that is called when an object property value is read or written
* This property holds a string corresponding to the MM label of a PyDevice object.
* When the property is set, look up the corresponding Python object and store a reference to that object in the Python property.
* Note: Unfortunately, we cannot use MM's built in object map because we cannot cast a MM::Device to a CPyDevice object because of the Curiously Recurring Template pattern used by CGenericBase. Therefore, we have to keep a list of devices ourselves.
*/
/**
* Callback that is called when an object property value is read or written
* This property holds a string corresponding to the MM label of a PyDevice object.
* When the property is set, look up the corresponding Python object and store a reference to that object in the Python property.
* Note: Unfortunately, we cannot use MM's built in object map because we cannot cast a MM::Device to a CPyDevice object because of the Curiously Recurring Template pattern used by CGenericBase. Therefore, we have to keep a list of devices ourselves.
*/
int PythonBridge::OnObjectProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int PythonBridge::Call(const PyObj& callable, PyObj& return_value) const noexcept {
    PyLock lock;
    return_value = PyObj(PyObject_CallNoArgs(callable));
    return CheckError();
}






/// Helper functions for finding the Python installation folder
///
/// 
bool PythonBridge::HasPython(const fs::path& path) noexcept {
    std::ifstream test(path / "python3.dll");
    return test.good();
}

/// Tries to locate the Python library. 
/// If Python is already initialized, returns the path used in the previous initialization.
/// If Python could not be found, returns an empty string
fs::path PythonBridge::FindPython() noexcept {
    if (g_initializedInterpreter)
        return g_PythonHome;

    std::string home_text;
    std::stringstream path(getenv("PATH"));
    while (std::getline(path, home_text, ';') && !home_text.empty()) {
        auto home = fs::path(home_text);
        auto home_lib1 = home.parent_path() / "lib";
        auto home_lib2 = home / "lib";
        if (HasPython(home))
            return home;
        if (HasPython(home_lib1))
            return home_lib1;
        if (HasPython(home_lib2))
            return home_lib2;
    }
    return string();
}


