
#include "pch.h"
#include "PythonBridge.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
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
std::unordered_map<string, PyObject*> PythonBridge::g_Devices;
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
        "import traceback\n"
        "code = open('" << scriptPath.generic_string() << "')\n"
        "exec(code.read())\n"
        "code.close()\n"
        "device = " << pythonClass << "()\n"
        "options = [p for p in type(device).__dict__.items() if isinstance(p[1], base_property)]";

    auto scope = PyObj(PyDict_New()); // create a scope to execute the scripts in
    auto bootstrap_result = PyObj(PyRun_String(bootstrap.str().c_str(), Py_file_input, scope, scope));
    object_ = PyObj::Borrow(PyDict_GetItemString(scope, "device"));
    options_ = PyObj::Borrow(PyDict_GetItemString(scope, "options"));
    intPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "int_property"));
    floatPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "float_property"));
    stringPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "string_property"));
    objectPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "object_property"));
    return CheckError(); // check python errors one more time just to be sure
}

/**
 * Destroys the Python object by removing all references to it
 * If there are other references
 * @return 
*/
int PythonBridge::Destruct() noexcept {
    PyLock lock;
    object_.Clear();
    options_.Clear();
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
int PythonBridge::SetProperty(const char* name, long value) noexcept {
    PyLock lock;
    PyObject_SetAttrString(object_, name, PyLong_FromLong(value));
    return CheckError();
}

int PythonBridge::SetProperty(const char* name, double value) noexcept {
    PyLock lock;
    PyObject_SetAttrString(object_, name, PyFloat_FromDouble(value));
    return CheckError();
}

int PythonBridge::SetProperty(const char* name, const string& value) noexcept {
    PyLock lock;
    PyObject_SetAttrString(object_, name, PyUnicode_FromString(value.c_str()));
    return CheckError();
}

int PythonBridge::SetProperty(const char* name, PyObject* value) noexcept {
    PyLock lock;
    PyObject_SetAttrString(object_, name, value);
    return CheckError();
}


/**
 * Reads the value of an object attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a long integer
*/
int PythonBridge::Get(PyObject* object, const char* name, PyObj& value) const noexcept {
    PyLock lock;
    value = PyObj(PyObject_GetAttrString(object, name));
    return CheckError();
}

/**
 * Reads the value of an integer attribute
 * 
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a long integer
*/
int PythonBridge::Get(PyObject* object, const char* name, long& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyLong_AsLong(attr);
    return CheckError();
}

/**
 * Reads the value of a float attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a double
*/
int PythonBridge::Get(PyObject* object, const char* name, double& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyFloat_AsDouble(attr);
    return CheckError();
}

/**
 * Reads the value of a string attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or does not hold a string
*/
int PythonBridge::Get(PyObject* object, const char* name, std::string& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyUTF8(attr);
    return CheckError();
}

int PythonBridge::Call(const PyObj& callable, PyObj& return_value) const noexcept {
    PyLock lock;
    return_value = PyObj(PyObject_CallNoArgs(callable));
    return return_value ? DEVICE_OK : ERR_PYTHON_EXCEPTION;
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

/**
 * @brief Converts a Python string object to an UTF-8 encoded c++ string
 * @param obj 
 * @return 
*/
string PythonBridge::PyUTF8(PyObject* obj) {
    PyLock lock;
    if (!obj)
        return string();
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s)
        return string();
    return s;
}

