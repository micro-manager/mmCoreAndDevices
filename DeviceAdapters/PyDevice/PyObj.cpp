#include "PyObj.h"

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
 * Queries the Python error state to get additional information about the error, and resets the error state.
 *
*/
void PyObj::ReportError() {
    // prevent infinite recusion if an error happens in the CheckError function itself
    static bool reentrant = false;
    
    // assert(PyLock.IsLocked()); // we can only call Python functions if we hold the GIL
    if (reentrant || !PyErr_Occurred())
        return;
    reentrant = true;

    auto msg = string("Python error.");
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    if (type) {
        msg += PyObj(PyObject_Str(type)).as<string>();
        msg += " : ";
    }
    if (value)
        msg += PyObj(PyObject_Str(value)).as<string>();

    if (traceback) {
        auto scope = PyObj(PyDict_New());
        PyDict_SetItemString(scope, "_current_tb", traceback);
        auto import_result = PyObj(PyRun_String("import traceback\n", Py_file_input, scope, scope));
        auto trace = PyObj(PyRun_String("''.join(traceback.format_tb(_current_tb))", Py_eval_input, scope, scope));
        msg += trace.as<string>();
    }
    PyErr_Restore(type, value, traceback);
    PyErr_Clear();
    g_errorMessage += msg + '\n';
    reentrant = false;
}



/// Helper functions for finding the Python installation folder
///
/// 
bool HasPython(const fs::path& path) noexcept {
    std::ifstream test(path / "python3.dll");
    return test.good();
}

/// Tries to locate the Python library. 
/// If Python is already initialized, returns the path used in the previous initialization.
/// If Python could not be found, returns an empty string
fs::path PyObj::FindPython() noexcept {
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

