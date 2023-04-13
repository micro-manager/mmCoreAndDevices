#include <string>
#include <functional>
#include <filesystem>
#include <vector>
#include "MMDeviceConstants.h"

namespace fs = std::filesystem;

#define ERR_PYTHON_NOT_FOUND 101
#define ERR_PYTHON_PATH_CONFLICT 102
#define ERR_PYTHON_SCRIPT_NOT_FOUND 103
#define ERR_PYTHON_CLASS_NOT_FOUND 104
#define ERR_PYTHON_EXCEPTION 105
#define ERR_PYTHON_NO_INFO 106

#pragma once
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#define _DEBUG
#else
#include <Python.h>
#endif
using std::string;
using std::wstring;

class PyObj {
    PyObject* _p;
public:
    PyObj() : _p(nullptr) {
    }
    explicit PyObj(PyObject* obj) : _p(obj) {
        Py_XINCREF(_p);
    }
    PyObj(const PyObj& other) : _p(other) {
        Py_XINCREF(_p);
    }
    ~PyObj() {
        Py_XDECREF(_p);
    }
    operator PyObject* () const { 
        return _p;
    }
    PyObj& operator = (const PyObj& other) {
        Py_XDECREF(_p);
        _p = other;
        Py_XINCREF(_p);
        return *this;
    }
};

struct PythonProperty {
    string name;
    MM::PropertyType type;
};

class PythonBridge
{
    static unsigned int g_ActiveDeviceCount;
    static wstring g_PythonHome;
    static PyObj g_Module;
    PyObj _object;
    PyObj _options;
    PyObj _intPropertyType;
    PyObj _floatPropertyType;
    PyObj _stringPropertyType;

    std::function<void(const char*)> _errorCallback;
public:
    PythonBridge();
    int Construct(const char* pythonHome, const char* pythonScript, const char* pythonClass);
    int Destruct();
    void SetErrorCallback(const std::function<void(const char*)>& callback) {
        _errorCallback = callback;
    }
    int SetProperty(const string& name, long value);
    int SetProperty(const string& name, double value);
    int SetProperty(const string& name, const char* value);
    std::vector<PythonProperty> EnumerateProperties();
    static bool PythonActive() {
        return g_ActiveDeviceCount > 0;
    }
    static fs::path FindPython();


    //static string DefaultPluginPath();
private:
    static bool HasPython(const fs::path& path);
    int PythonError();
    int ConstructInternal(const char* pythonScript, const char* pythonClass);
    static string PyUTF8(PyObject* obj);
};


// helper functions for converting between Python and MM strings
static string WStringToString(const wstring& w);
static wstring StringToWString(const string& a);
