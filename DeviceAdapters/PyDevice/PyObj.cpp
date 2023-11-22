#include "pch.h"
#include "PyObj.h"
#include <numpy/arrayobject.h>



PyObj PyObj::g_unit_ms;
PyObj PyObj::g_unit_um;
PyObj PyObj::g_traceback_to_string;
PyObj PyObj::g_set_path;
PyObj PyObj::g_scan_devices;
PyObj PyObj::g_main_module;
PyObj PyObj::g_global_scope;
PyThreadState* PyObj::g_threadState = nullptr;

/**
* Takes a new reference and wraps it into a PyObj smart pointer.
*
* This function does not increase the reference count of the object (also see Borrow). The reference count is decreased when the PyObj smart pointer is destroyed (e.g. when it goes out of scope).
*
*/
PyObj::PyObj(PyObject* obj) : p_(obj) {
    if (!obj)
        ReportError();
}

string PyObj::g_errorMessage;


/**
@brief Initializes the Python interpreter.
@param module_path Additional folders to search for modules, separated by ';'.
@return true on success, false on failure (the g_errorMessage field will be set).
*/
bool PyObj::InitializeInterpreter(const string& module_path) noexcept
{
    // If the interpreter is already initialized, only change the search path
    if (g_threadState != nullptr) {
        g_set_path.Call(PyObj(module_path));
        return true;
    }
    
    auto path = fs::path();// python_home;
    
    //todo: windows specific
    auto env_path = _wgetenv(L"PYTHONHOME");
    if (env_path && env_path[0] != 0) {
        path = env_path;
    }
    else {
        // fallback: use python3.dll location
        HMODULE hModule = GetModuleHandle(L"python3.dll");
        TCHAR dllPath[_MAX_PATH];
        GetModuleFileName(hModule, dllPath, _MAX_PATH);
        path = fs::path(dllPath).parent_path();
    }
    Py_SetPythonHome(path.generic_wstring().c_str());

    // set the module search path if it is specified in an environmental variable
    auto base_module_path = _wgetenv(L"PYTHONPATH");
    if (base_module_path && base_module_path[0] != 0) {
        Py_SetPath(base_module_path);
    }

    Py_InitializeEx(0); // Python may cause a crash here (all exit()) if the runtime cannot be initialized. There seems to be nothing we can do about this in the LIMITED api.
    _import_array(); // initialize numpy. We don't use import_array (without _) because it hides any error message that may occur.


    // allow multi threading and store the thread state (global interpreter lock).
    // note: savethread releases the lock.
    g_threadState = PyEval_SaveThread();

    // run the bootstrapping script
    const char* bootstrap;
    #include "bootstrap.py"
            
    PyLock lock;
    g_main_module = PyObj(PyImport_AddModule("__main__"));
    g_global_scope = PyObj(PyModule_GetDict(g_main_module));
    g_global_scope.SetDictItem("_EXTRA_SEARCH_PATH", module_path);
    if (!RunScript(&bootstrap[1], "bootstrap.py", g_global_scope))
        return false;

    // get the ms and um units
    g_unit_ms = g_global_scope.GetDictItem("unit_ms");
    g_unit_um = g_global_scope.GetDictItem("unit_um");
    g_traceback_to_string = g_global_scope.GetDictItem("traceback_to_string");
    g_set_path = g_global_scope.GetDictItem("set_path");
    g_scan_devices = g_global_scope.GetDictItem("scan_devices");
    return ReportError();
}

/**
 * @brief Compiles and executes the Python code
 * @param code Python source code 
 * @param file_name Value of __file__. Also used in tracebacks
 * @param locals Dictionary object that holds the local variables of the script. Can be used to 'return' values from the script
 * @return true on success, false on failure (g_errorMessage will be set)
*/
bool PyObj::RunScript(const string& code, const string& file_name, const PyObj& locals) noexcept {
    PyLock lock;
    auto bootstrap_code = PyObj(Py_CompileString(code.c_str(), file_name.c_str(), Py_file_input));
    if (!bootstrap_code)
        return false;
    return PyObj(PyEval_EvalCode(bootstrap_code, g_global_scope, locals)); // Py_None on success (->true), NULL on failure (->false)
}


/**
 * Queries the Python error state to get additional information about the error, and resets the error state.
 * Returns false if an error occurred
 *
*/
bool PyObj::ReportError() {
    PyLock lock; // todo: should not be needed. If we don't hold the GIL, ReportError is undefined behavior in a multi-threaded context?
    if (!PyErr_Occurred())
        return true;
    
    // prevent infinite recusion if an error happens in the CheckError function itself
    static bool reentrant = false;
    if (reentrant) {
        PyErr_Clear();
        return false;
    }
    reentrant = true;

    auto msg = string("Python error.");
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    if (type) {
        msg += Borrow(type).as<string>();
        msg += " : ";
    }
    if (value)
        msg += Borrow(value).as<string>();

    if (traceback)
        msg += g_traceback_to_string.Call(Borrow(traceback)).as<string>();
    
    PyErr_Restore(type, value, traceback);
    PyErr_Clear();
    g_errorMessage += msg + '\n';
    reentrant = false;
    return false;
}

