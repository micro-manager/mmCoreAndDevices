#include "pch.h"
#include "PyObj.h"
#include <numpy/arrayobject.h>



PyObj PyObj::g_unit_ms;
PyObj PyObj::g_unit_um;
PyObj PyObj::g_traceback_to_string;
PyObj PyObj::g_execute_script;
fs::path PyObj::g_python_home;
PyThreadState* PyObj::g_threadState = nullptr;

/**
* Takes a new reference and wraps it into a PyObj smart pointer.
*
* This function does not increase the reference count of the object (also see PyObj::Borrow). The reference count is decreased when the PyObj smart pointer is destroyed (e.g. when it goes out of scope).
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
        PyObj::g_python_home = Py_GetPythonHome();

        // run the bootstrapping script
        const char* bootstrap;
        #include "bootstrap.py"
            
        PyLock lock;
        auto scope = PyObj(PyDict_New()); // create a scope to execute the scripts in
        auto test = Py_CompileString("test=1", "test.py", Py_single_input);
        auto bootstrap_code = PyObj(Py_CompileString(&bootstrap[1], "bootstrap.py", Py_file_input));
        auto bootstrap_result = PyObj(PyEval_EvalCode(bootstrap_code, scope, scope));
        if (!bootstrap_result)
            return PyObj::ReportError();

        // get the ms and um units
        g_unit_ms = PyObj::Borrow(PyDict_GetItemString(scope, "unit_ms"));
        g_unit_um = PyObj::Borrow(PyDict_GetItemString(scope, "unit_um"));
        g_traceback_to_string = PyObj::Borrow(PyDict_GetItemString(scope, "traceback_to_string"));
        g_execute_script = PyObj::Borrow(PyDict_GetItemString(scope, "execute_script"));
    }
    else {
        if (!python_home.empty() && python_home != PyObj::g_python_home) {
            return ERR_PYTHON_MULTIPLE_INTERPRETERS;
        }
    }
    return 0;
}

/**
 * Loads the Python script and creates a device object
 * @param pythonScript path of the .py script file
 * @param pythonClass name of the Python class to create an instance of
 * @return MM return code
*/
PyObj PyObj::RunScript(const fs::path& script_path) noexcept {
    PyLock lock;
    auto path = PyObj(script_path.generic_u8string().c_str());
    return PyObj::g_execute_script.Call(path);
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

    if (traceback) {
        auto trace = PyObj::g_traceback_to_string.Call(PyObj::Borrow(traceback));
//        PyDict_SetItemString(scope, "_current_tb", traceback);
//        auto import_result = PyObj(PyRun_String("import traceback\n", Py_file_input, scope, scope));
//        auto trace = PyObj(PyRun_String("''.join(traceback.format_tb(_current_tb))", Py_eval_input, scope, scope));
        msg += trace.as<string>();
    }
    PyErr_Restore(type, value, traceback);
    PyErr_Clear();
    g_errorMessage += msg + '\n';
    reentrant = false;
    return false;
}


/// Tries to locate the Python library. 
/// If Python is already initialized, returns the path used in the previous initialization.
/// If Python could not be found, returns an empty string
/*fs::path PyObj::FindPython() noexcept {
    std::string home_text;
    std::stringstream path(getenv("PATH"));
    while (std::getline(path, home_text, ';') && !home_text.empty()) {
        auto home = fs::path(home_text);
        auto home_lib1 = home.parent_path() / "lib";
        auto home_lib2 = home / "lib";
        if (FileExists(home / "python3.dll"))
            return home;
        if (FileExists(home_lib1 / "python3.dll"))
            return home_lib1;
        if (FileExists(home_lib2 / "python3.dll"))
            return home_lib2;
    }
    return string();
}*/

