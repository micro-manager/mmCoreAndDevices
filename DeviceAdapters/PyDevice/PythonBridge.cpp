
#include "pch.h"
#include "PythonBridge.h"
#include <fstream>
#include <sstream>

// see https://numpy.org/doc/stable/reference/c-api/array.html#c.import_array
#include <numpy/arrayobject.h>


bool PythonBridge::g_initializedInterpreter = false;
PyThreadState* PythonBridge::g_threadState = nullptr; // used to store the thread state when releasing the global interpreter lock for the first time. Currently this value is never used, but is may be used to properly shut down the Python interpreter.
fs::path PythonBridge::g_PythonHome;


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
    // Check if Python already initialized
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

    _import_array(); // initialize numpy
    if (PyErr_Occurred())
        return PythonError();

    g_threadState = PyEval_SaveThread(); // allow multi threading
    g_PythonHome = homePath;
    g_initializedInterpreter = true;
    return DEVICE_OK;
}

// Load python script
// This is done by constructing a python loader script and executing it.
int PythonBridge::ConstructPythonObject(const char* pythonScript, const char* pythonClass) noexcept {
    //PyLock lock (already locked by caller)
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

    try {
        module_ = PyObj(PyDict_New()); // create a scope to execute the scripts in

        auto bootstrap_result = PyObj(PyRun_String(bootstrap.str().c_str(), Py_file_input, module_, module_));
        object_ = PyObj(PyDict_GetItemString(module_, "device"));
        options_ = PyObj(PyDict_GetItemString(module_, "options"));
        intPropertyType_ = PyObj(PyDict_GetItemString(module_, "int_property"));
        floatPropertyType_ = PyObj(PyDict_GetItemString(module_, "float_property"));
        stringPropertyType_ = PyObj(PyDict_GetItemString(module_, "string_property"));
    }
    catch (PyObj::PythonException) {
        return PythonError();
    }
    return DEVICE_OK;
}

int PythonBridge::Destruct() noexcept {
    PyLock lock;
    object_.Clear();
    options_.Clear();
    intPropertyType_.Clear();
    floatPropertyType_.Clear();
    stringPropertyType_.Clear();
    module_.Clear();
    initialized_ = false;
    return DEVICE_OK;
}

int PythonBridge::SetProperty(const char* name, long value) noexcept {
    PyLock lock;
    return PyObject_SetAttrString(object_, name, PyLong_FromLong(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const char* name, double value) noexcept {
    PyLock lock;
    return PyObject_SetAttrString(object_, name, PyFloat_FromDouble(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const char* name, const string& value) noexcept {
    PyLock lock;
    return PyObject_SetAttrString(object_, name, PyUnicode_FromString(value.c_str())) == 0 ? DEVICE_OK : PythonError();
}


int PythonBridge::GetProperty(const char* name, long &value) const noexcept {
    return GetInt(object_, name, value);
}

int PythonBridge::GetProperty(const char* name, double& value) const noexcept {
    return GetFloat(object_, name, value);
}

int PythonBridge::GetProperty(const char* name, string& value) const noexcept {
    return GetString(object_, name, value);
}

int PythonBridge::GetProperty(const char* name, PyObj& value) const noexcept {
    return GetAttr(object_, name, value);
}

/**
 * Reads the value of an object attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a long integer
*/
int PythonBridge::GetAttr(PyObject* object, const char* name, PyObj& value) const noexcept {
    PyLock lock;
    value = PyObj(PyObject_GetAttrString(object, name));
    return (PyErr_Occurred() || !value) ? PythonError() : DEVICE_OK;
}


/**
 * Reads the value of an integer attribute
 * 
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a long integer
*/
int PythonBridge::GetInt(PyObject* object, const char* name, long& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyLong_AsLong(attr);
    return PyErr_Occurred() ? PythonError() : DEVICE_OK;
}

/**
 * Reads the value of a float attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or could not be converted to a double
*/
int PythonBridge::GetFloat(PyObject* object, const char* name, double& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyFloat_AsDouble(attr);
    return PyErr_Occurred() ? PythonError() : DEVICE_OK;
}

/**
 * Reads the value of a string attribute
 *
 * @param object Python object holding the attribute
 * @param name name of the attribute
 * @param value output that will hold the value on success
 * @return MM error code, DEVICE_OK on success, ERR_PYTHON if the attribute was missing or does not hold a string
*/
int PythonBridge::GetString(PyObject* object, const char* name, std::string& value) const noexcept {
    PyLock lock;
    auto attr = PyObject_GetAttrString(object, name);
    if (attr)
        value = PyUTF8(attr);
    return PyErr_Occurred() ? PythonError() : DEVICE_OK;
}

int PythonBridge::PythonError() const {
    PyLock lock;
    if (!PyErr_Occurred())
        return ERR_PYTHON_NO_INFO;
    if (errorCallback_) {
        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        auto msg = string("Python error.");
        if (type) {
            msg += PyUTF8(PyObj(PyObject_Str(type)));
            msg += " : ";
        }
        if (value)
            msg += PyUTF8(PyObj(PyObject_Str(value)));
        
        if (traceback) {
            try {
                PyDict_SetItemString(module_, "_current_tb", traceback);
                auto trace = PyObj(PyRun_String("''.join(traceback.format_tb(_current_tb))", Py_eval_input, module_, module_));
                msg += PyUTF8(trace);
            }
            catch (PyObj::PythonException e) {
                msg += "[could not get stack trace]";
            }
        }
        errorCallback_(msg.c_str());
        PyErr_Restore(type, value, traceback);
        PyErr_Clear();
        return ERR_PYTHON_EXCEPTION;
    }
    else {
        PyErr_Clear();
        return ERR_PYTHON_NO_INFO;
    }    
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


string PythonBridge::PyUTF8(PyObject* obj) {
    PyLock lock;
    if (!obj)
        return string();
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s)
        return string();
    return s;
}

