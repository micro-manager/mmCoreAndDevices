#include "PythonBridge.h"
#include <fstream>
#include <sstream>
#include <windows.h>
#include <numpy/arrayobject.h>

unsigned int PythonBridge::g_ActiveDeviceCount = 0;
fs::path PythonBridge::g_PythonHome;
PyObj PythonBridge::g_Module;



// set python path. The folder must contain the python dll. If Python is already initialized, the path must be the same
// as the path that was used for initializing it. So, multiple deviced must have the same python install path.
int PythonBridge::InitializeInterpreter(const char* pythonHome)
{
    // Initialize Python interperter
    auto homePath = fs::path(pythonHome);
    if (PythonActive()) {
        if (homePath != fs::path(g_PythonHome))
            return ERR_PYTHON_PATH_CONFLICT;
    }
    else {
        if (!HasPython(pythonHome))
            return ERR_PYTHON_NOT_FOUND;
        g_PythonHome = homePath;
        Py_SetPythonHome(homePath.generic_wstring().c_str());
        Py_Initialize();
        g_Module = PyObj(PyDict_New()); // create a global scope to execute the scripts in
        import_array(); // initialize numpy
        threadState_ = PyEval_SaveThread(); // allow multi threading
    }
    return DEVICE_OK;
}

int PythonBridge::ConstructPythonObject(const char* pythonScript, const char* pythonClass) {

    // Load python script
    // This is done by constructing a python loader script and executing it.
    auto scriptPath = fs::absolute(fs::path(pythonScript));
    auto bootstrap = std::stringstream();
    bootstrap <<
        "import numpy as np\n"
        "code = open('" << scriptPath.generic_string() << "')\n"
        "exec(code.read())\n"
        "code.close()\n"
        "device = " << pythonClass << "()\n"
        "options = [p for p in type(device).__dict__.items() if isinstance(p[1], base_property)]";

    try {
        auto bootstrap_result = PyObj(PyRun_String(bootstrap.str().c_str(), Py_file_input, g_Module, g_Module));
        _object = PyObj(PyDict_GetItemString(g_Module, "device"));
        _options = PyObj(PyDict_GetItemString(g_Module, "options"));
        _intPropertyType = PyObj(PyDict_GetItemString(g_Module, "int_property"));
        _floatPropertyType = PyObj(PyDict_GetItemString(g_Module, "float_property"));
        _stringPropertyType = PyObj(PyDict_GetItemString(g_Module, "string_property"));
        if (!PyArray_API)
            import_array(); // initialize numpy again!
    }
    catch (PyObj::NullPointerException) {
        return PythonError();
    }
    return DEVICE_OK;
}

int PythonBridge::Destruct() {
    PyLock lock;
    _object.Clear();
    _options.Clear();
    _intPropertyType.Clear();
    _floatPropertyType.Clear();
    _stringPropertyType.Clear();
    if (initialized_) {
        initialized_ = false;
        g_ActiveDeviceCount--;
//        if (!g_ActiveDeviceCount)
//            DeinitializeInterpreter();
    }
    return DEVICE_OK;
}

int PythonBridge::SetProperty(const char* name, long value) {
    PyLock lock;
    return PyObject_SetAttrString(_object, name, PyLong_FromLong(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const char* name, double value) {
    PyLock lock;
    return PyObject_SetAttrString(_object, name, PyFloat_FromDouble(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const char* name, const string& value) {
    PyLock lock;
    return PyObject_SetAttrString(_object, name, PyUnicode_FromString(value.c_str())) == 0 ? DEVICE_OK : PythonError();
}

PyObj PythonBridge::GetAttr(PyObject* object, const char* string) {
    PyLock lock;
    return PyObj(PyObject_GetAttrString(object, string));
}

int PythonBridge::GetProperty(const char* name, long &value) const {
    value = GetInt(_object, name);
    return DEVICE_OK;
}

int PythonBridge::GetProperty(const char* name, double& value) const {
    value = GetFloat(_object, name);
    return DEVICE_OK;
}

int PythonBridge::GetProperty(const char* name, string& value) const {
    value = GetString(_object, name);
    return DEVICE_OK;
}

PyObj PythonBridge::GetProperty(const char* name) const {
    return GetAttr(_object, name);
}


long PythonBridge::GetInt(PyObject* object, const char* string) {
    PyLock lock;
    return PyLong_AsLong(GetAttr(object, string));
}

double PythonBridge::GetFloat(PyObject* object, const char* string) {
    PyLock lock;
    return PyFloat_AsDouble(GetAttr(object, string));
}

string PythonBridge::GetString(PyObject* object, const char* string) {
    PyLock lock;
    return PyUTF8(GetAttr(object, string));
}


int PythonBridge::PythonError() const {
    PyLock lock;
    if (!PyErr_Occurred())
        return ERR_PYTHON_NO_INFO;
    if (_errorCallback) {
        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);
        auto msg = string("Python error. ");
        if (type) {
            msg += PyUTF8(PyObj(PyObject_Str(type)));
            msg += " : ";
        }
        if (value)
            msg += PyUTF8(PyObj(PyObject_Str(value)));
        
        _errorCallback(msg.c_str());
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
bool PythonBridge::HasPython(const fs::path& path) {
    if (path.empty())
        return false;

    std::ifstream test(path / "python3.dll");
    return test.good();
}

/// Tries to locate the Python library. 
/// If Python is already initialized, returns the path used in the previous initialization.
/// If Python could not be found, returns an empty string
fs::path PythonBridge::FindPython() {
    if (PythonActive())
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

