#include "pch.h"
#include "PyObj.h"
#include <numpy/arrayobject.h>



PyObj PyObj::g_unit_ms;
PyObj PyObj::g_unit_um;
PyObj PyObj::g_traceback_to_string;
PyObj PyObj::g_add_to_path;
PyObj PyObj::g_scan_devices;
PyObj PyObj::g_main_module;
PyObj PyObj::g_global_scope;

fs::path PyObj::g_python_home;
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
@param venv Optional path to a venv virtual environment folder.
@return true on success, false on failure (the g_errorMessage field will be set).

If the interpreter is already initialized, this function checks if venv is either empty, or equal to the previously passed venv path. If not, anerror is reported. We cannot run multiple interpreters at once. Unfortunately, we cannot de - initialize one interpreter and start a new one. Thiswould require calling Py_FinalizeEx, and then Py_Initialize again.Unfortunatly, by the Python docs(https://docs.python.org/3/c-apiinit.html#c.Py_FinalizeEx), some extension modules(apparanetly including numpy) may not support this behavior, making Py_Finalize followed by Py_Initialize completely undefined behavior, and weird numpy - related crashes were seen when trying it.
*/
bool PyObj::InitializeInterpreter(const fs::path& python_home) noexcept
{
    // Initilialize Python interpreter, if not already done
    auto path = python_home;
    if (g_threadState == nullptr) {
        if (path.empty()) {
            // get from env variable
            auto env_path = getenv("PYTHONPATH");
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
        }
        Py_SetPythonHome(path.generic_wstring().c_str());

        Py_InitializeEx(0);
        _import_array(); // initialize numpy. We don't use import_array (without _) because it hides any error message that may occur.

        // allow multi threading and store the thread state (global interpreter lock).
        // note: savethread releases the lock.
        g_threadState = PyEval_SaveThread();
        g_python_home = Py_GetPythonHome();

        // run the bootstrapping script
        const char* bootstrap;
        #include "bootstrap.py"
            
        PyLock lock;
        g_main_module = PyObj(PyImport_AddModule("__main__"));
        g_global_scope = PyObj(PyModule_GetDict(g_main_module));
        if (!RunScript(&bootstrap[1], "bootstrap.py", g_global_scope))
            return false;

        // get the ms and um units
        g_unit_ms = Borrow(PyDict_GetItemString(g_global_scope, "unit_ms"));
        g_unit_um = Borrow(PyDict_GetItemString(g_global_scope, "unit_um"));
        g_traceback_to_string = Borrow(PyDict_GetItemString(g_global_scope, "traceback_to_string"));
        g_add_to_path = Borrow(PyDict_GetItemString(g_global_scope, "add_to_path"));
        g_scan_devices = Borrow(PyDict_GetItemString(g_global_scope, "scan_devices"));
        return ReportError();
    }
    else {
        if (!python_home.empty() && python_home != g_python_home) {
            g_errorMessage += "A different Python interpreter was already started. "
                "Due to limitations in the Python runtime, it is not possible to switch virtual environments. Please fix the Python library path, or leave it blank.\n"
                "If you want to change the Python interpreter or virtual environment, you will have to restart Micro-Manager with a configuration of (none), and rebuild the configuration using the new interpreter.\n";
            return false;
        } else
            return true;
    }
}

/**
 * @brief Compiles and executes the Python code
 * @param code Python source code 
 * @param file_name Value of __file__. Also used in tracebacks
 * @param locals Dictionary object that holds the local variables of the script. Can be used to 'return' values from the script
 * @return true on success, false on failure (g_errorMessage will be set)
*/
bool PyObj::RunScript(const char* code, const char* file_name, const PyObj& locals) noexcept {
    PyLock lock;
    auto bootstrap_code = PyObj(Py_CompileString(code, file_name, Py_file_input));
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
    // prevent infinite recusion if an error happens in the CheckError function itself
    static bool reentrant = false;
    
    // assert(PyLock.IsLocked()); // we can only call Python functions if we hold the GIL
    if (reentrant)
        return false;
    PyLock lock;
    if (!PyErr_Occurred())
        return true;
    reentrant = true;

    auto msg = string("Python error.");
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    if (type) {
        msg += PyObj(type).as<string>();
        msg += " : ";
    }
    if (value)
        msg += PyObj(value).as<string>();

    if (traceback)
        msg += g_traceback_to_string.Call(Borrow(traceback)).as<string>();

    PyErr_Restore(type, value, traceback);
    PyErr_Clear();
    g_errorMessage += msg + '\n';
    reentrant = false;
    return false;
}

