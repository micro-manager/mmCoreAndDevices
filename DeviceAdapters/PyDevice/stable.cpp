#include "pch.h"

PyObject* Py_None = nullptr;
PyObject* Py_True = nullptr;
PyObject* Py_False = nullptr;

bool LoadPythonData()
{
    // Load the DLL
    HMODULE hModule = LoadLibrary(L"python39.dll");
    if (hModule == NULL) {
        //std::cerr << "Failed to load DLL" << std::endl;
        return false;
    }

    // Get the function
    Py_None = (PyObject*)GetProcAddress(hModule, "_Py_NoneStruct");
    Py_True = (PyObject*)GetProcAddress(hModule, "_Py_TrueStruct");
    Py_False = (PyObject*)GetProcAddress(hModule, "_Py_FalseStruct");
    if (Py_None == nullptr || Py_False == nullptr || Py_True == nullptr) {
        // std::cerr << "Failed to find GetPerson function" << std::endl;
        FreeLibrary(hModule);
        return false;
    }

    // Free the library
//    FreeLibrary(hModule);
    return true;
}