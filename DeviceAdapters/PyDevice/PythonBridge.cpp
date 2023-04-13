#include "PythonBridge.h"
#include <fstream>
#include <sstream>
#include <windows.h>
#include "MMDeviceConstants.h"

unsigned int PythonBridge::g_ActiveDeviceCount = 0;
wstring PythonBridge::g_PythonHome;
PyObj PythonBridge::g_Module;

PythonBridge::PythonBridge() {
}


// set python path. The folder must contain the python dll. If Python is already initialized, the path must be the same
        // as the path that was used for initializing it. So, multiple deviced must have the same python install path.
int PythonBridge::Construct(const char* pythonHome, const char* pythonScript, const char* pythonClass)
{
    // Initialize Python interperter
    auto homePath = fs::path(StringToWString(pythonHome));
    if (PythonActive()) {
        if (homePath != fs::path(g_PythonHome))
            return ERR_PYTHON_PATH_CONFLICT;
    }
    else {
        if (!HasPython(pythonHome))
            return ERR_PYTHON_NOT_FOUND;
        g_PythonHome = homePath;
        Py_SetPythonHome(g_PythonHome.c_str());
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
    auto option_count = PyList_Size(_options);
    for (Py_ssize_t i = 0; i < option_count; i++) {
        auto key_value = PyList_GetItem(_options, i); // note: borrowed reference, don't ref count (what a mess...)
        auto name = PyUTF8(PyTuple_GetItem(key_value, 0));
        if (name.empty())
            return PythonError();
        auto property = PyTuple_GetItem(key_value, 1);
    }
    PyObject_Dir(_options);
    return DEVICE_OK;
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
        return ERR_PYTHON_EXCEPTION;
    } else
        return ERR_PYTHON_NO_INFO;
    
    PyErr_Clear();
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
        return WStringToString(g_PythonHome);

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


string WStringToString(const wstring& w) {
    std::string converted;
    if (w.empty())
        return converted;

    auto resultLen = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    if (!resultLen)
        return "Error converting string";

    converted.resize(resultLen);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), converted.data(), resultLen, nullptr, nullptr);
    return converted;
}
string WStringToString(const wchar_t* w) {
    return w ? WStringToString(wstring(w)) : string();
}

wstring StringToWString(const string& a) {
    std::wstring converted;
    if (a.empty())
        return converted;

    auto resultLen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, a.c_str(), (int)a.size(), nullptr, 0);
    if (!resultLen)
        return L"Error converting string";

    converted.resize(resultLen);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, a.c_str(), (int)a.size(), converted.data(), resultLen);
    return converted;
}

string PythonBridge::PyUTF8(PyObject* obj) {
    if (!obj)
        return string();
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s)
        return string();
    return s;
}
