#include "PythonBridge.h"
#include <fstream>
#include <sstream>
#include <windows.h>

unsigned int PythonBridge::g_ActiveDeviceCount = 0;
fs::path PythonBridge::g_PythonHome;
PyObj PythonBridge::g_Module;

PythonBridge::PythonBridge() {
}


// set python path. The folder must contain the python dll. If Python is already initialized, the path must be the same
        // as the path that was used for initializing it. So, multiple deviced must have the same python install path.
int PythonBridge::Construct(const char* pythonHome, const char* pythonScript, const char* pythonClass)
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
    }
    g_ActiveDeviceCount++;
    auto result = ConstructInternal(pythonScript, pythonClass);
    if (result == DEVICE_OK)
        Destruct(); // if construction fails, clean up. If there are no more active devices, also de-initialize Python library
    return result;
}

int PythonBridge::ConstructInternal(const char* pythonScript, const char* pythonClass) {

    // Load python script
    // This is done by constructing a python loader script and executing it.
    auto scriptPath = fs::absolute(fs::path(pythonScript));
    auto bootstrap = std::stringstream();
    bootstrap <<
        "import traceback\n"
        "code = open('" << scriptPath.generic_string() << "')\n"
        "exec(code.read())\n"
        "code.close()\n"
        "device = " << pythonClass << "()\n"
        "options = [p for p in type(device).__dict__.items() if isinstance(p[1], base_property)]";


    auto code = bootstrap.str();
    auto result = PyObj(PyRun_String(code.c_str(), Py_file_input, g_Module, g_Module));
    if (!result)
        return PythonError();

    _object = PyObj(PyDict_GetItem(g_Module, PyObj(PyUnicode_FromString("device"))));
    _options = PyObj(PyDict_GetItem(g_Module, PyObj(PyUnicode_FromString("options"))));
    _intPropertyType = PyObj(PyDict_GetItem(g_Module, PyObj(PyUnicode_FromString("int_property"))));
    _floatPropertyType = PyObj(PyDict_GetItem(g_Module, PyObj(PyUnicode_FromString("float_property"))));
    _stringPropertyType = PyObj(PyDict_GetItem(g_Module, PyObj(PyUnicode_FromString("string_property"))));
    return DEVICE_OK;
}

std::vector<PythonProperty> PythonBridge::EnumerateProperties() {
    auto property_count = PyList_Size(_options);
    auto properties = std::vector<PythonProperty>();
    properties.reserve(property_count);

    for (Py_ssize_t i = 0; i < property_count; i++) {
        auto key_value = PyList_GetItem(_options, i); // note: borrowed reference, don't ref count (what a mess...)
        auto name = PyUTF8(PyTuple_GetItem(key_value, 0));
        if (name.empty())
            continue;
        auto property = PyTuple_GetItem(key_value, 1);
        auto lower = PyObject_HasAttrString(property, "min") ? PyFloat_AsDouble(PyObj(PyObject_GetAttrString(property, "min"))) : -std::numeric_limits<double>().infinity();
        auto upper = PyObject_HasAttrString(property, "max") ? PyFloat_AsDouble(PyObj(PyObject_GetAttrString(property, "max"))) : std::numeric_limits<double>().infinity();

        if (PyObject_IsInstance(property, _intPropertyType)) {
            properties.push_back({ name, MM::Integer, lower, upper });
        } else if (PyObject_IsInstance(property, _floatPropertyType)) {
            properties.push_back({ name, MM::Float, lower, upper });
        } else if (PyObject_IsInstance(property, _stringPropertyType)) {
            properties.push_back({ name, MM::String });
        }

    }
    return properties;
}

int PythonBridge::SetProperty(const string& name, long value) {
    return PyObject_SetAttrString(_object, name.c_str(), PyLong_FromLong(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const string& name, double value) {
    return PyObject_SetAttrString(_object, name.c_str(), PyFloat_FromDouble(value)) == 0 ? DEVICE_OK : PythonError();
}

int PythonBridge::SetProperty(const string& name, const char* value) {
    return PyObject_SetAttrString(_object, name.c_str(), PyUnicode_FromString(value)) == 0 ? DEVICE_OK : PythonError();
}


int PythonBridge::PythonError() {
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

/*
string PythonBridge::DefaultPluginPath()
{
    char path[MAX_PATH];
    HMODULE hm = nullptr;

    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&DefaultPluginPath, &hm) == 0)
        return string();
    if (GetModuleFileNameA(hm, path, sizeof(path)) == 0)
        return string();
    auto path_string = string(path);
    auto separator = path_string.rfind("\\");
    if (separator != string::npos)
        path_string = path_string.substr(0, separator);
    return path_string;
}*/



int PythonBridge::Destruct() {
    return DEVICE_OK;
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
    if (!obj)
        return string();
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s)
        return string();
    return s;
}
