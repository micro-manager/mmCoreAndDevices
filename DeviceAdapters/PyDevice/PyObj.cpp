#include "pch.h"
#include "PyObj.h"
//#include <numpy/arrayobject.h>

PyObj PyObj::g_traceback_to_string;
PyObj PyObj::g_load_devices;
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
@return true on success, false on failure (the g_errorMessage field will be set).
*/
bool PyObj::InitializeInterpreter() noexcept
{
    if (!Py_IsInitialized()) {
        // Start the Python interperter if it is not running yet
        //todo: windows specific
        auto path = fs::path();// python_home;
        auto env_path = _wgetenv(L"PYTHONHOME");
        if (env_path && env_path[0] != 0) {
            path = env_path;
        }
        else {
            // fallback: use python3.dll location
            HMODULE hModule = GetModuleHandle(L"python39.dll");
            TCHAR dllPath[_MAX_PATH];
            GetModuleFileName(hModule, dllPath, _MAX_PATH);
            path = fs::path(dllPath).parent_path();
        }
        // To initialize Python using the stable API, some functions (SetPythonHome, SetPath, InitializeEx) are needed.
        // Strangely, these functions are deprecated since Python 3.11 and no alternatives are provided.
        // As a workaround, suppress the warning.
#pragma warning(push)
#pragma warning(disable: 4996)
        Py_SetPythonHome(path.generic_wstring().c_str());

        // set the module search path if it is specified in an environmental variable
        // note that additional search paths can be specified by the user. These paths are only used during the 'load_script' call.
        auto base_module_path = _wgetenv(L"PYTHONPATH");
        if (base_module_path && base_module_path[0] != 0)
            Py_SetPath(base_module_path);

        Py_InitializeEx(0); // Python may cause a crash here (all exit()) if the runtime cannot be initialized. There seems to be nothing we can do about this in the LIMITED api.

#pragma warning(pop)
        // allow multi threading and store the thread state (global interpreter lock).
        // note: savethread releases the GIL lock we currently have.
        g_threadState = PyEval_SaveThread();
    }
    // If a Python interpreter is already running (this also happens when running from pymmcore), don't start a new interpreter again
    LoadPythonData();

    // run the bootstrapping script
    const char* bootstrap;
    #include "bootstrap.py"
            
    PyLock lock;
    auto module = PyObj(PyModule_New("pydevice"));
    // Set properties on the new module object
    if (PyModule_AddStringConstant(module, "__file__", "pydevice") != 0)
        return ReportError();

    g_global_scope = PyObj::Borrow(PyModule_GetDict(module));   // Returns a borrowed reference: no need to Py_DECREF() it once we are done
    auto builtins = PyObj::Borrow(PyEval_GetBuiltins());  // Returns a borrowed reference: no need to Py_DECREF() it once we are done
    if (PyDict_SetItemString(g_global_scope, "__builtins__", builtins) != 0)
        return ReportError();

    auto bootstrap_code = PyObj(Py_CompileString(&bootstrap[1], "bootstrap.py", Py_file_input));
    if (!bootstrap_code)
        return ReportError();

    if (PyEval_EvalCode(bootstrap_code, g_global_scope, g_global_scope) == nullptr)
        return ReportError();

    // get the um unit for use with stages
    g_traceback_to_string = g_global_scope.GetDictItem("traceback_to_string");
    g_load_devices = g_global_scope.GetDictItem("load_devices");
    return ReportError();
}

/**
 * @brief Clears all referencences to Python objects.
 * Note, this does _not_ call Py_Finalize() because deinitializing/initializing Python multiple times is undefined behavior.
 * Instead, we clean up as much as we can, making sure that this dll does not hold any refcount anymore.
*/
void PyObj::DeinitializeInterpreter() noexcept
{
    PyLock lock;
    g_traceback_to_string.Clear();
    g_load_devices.Clear();
    g_main_module.Clear();
    g_global_scope.Clear();
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
        msg += "\n" + g_traceback_to_string.Call(Borrow(traceback)).as<string>();
    
    PyErr_Restore(type, value, traceback);
    PyErr_Clear();
    g_errorMessage += msg + '\n';
    reentrant = false;
    return false;
}

