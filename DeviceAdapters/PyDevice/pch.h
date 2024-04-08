#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <functional>
#include <filesystem>
#pragma warning(disable: 5040) // disable warning we get because we are using C++17 for compilation.
#include <DeviceBase.h> // all MM includes
namespace fs = std::filesystem;
using std::string;
using std::function;
using std::vector;
using std::tuple;
using std::map;
#ifdef _WIN32 
#include <Windows.h>
#include <commdlg.h>
#endif

/// Helper function for checking if a file exists
inline bool FileExists(const fs::path& path) noexcept {
    std::ifstream test(path);
    return test.good();
}


// Use the limited Python API. Not all functions are available, but the advantage is that we can use python3.dll, which will be available
// on any system with python 3.2 or higher installed (and findable, meaning python3.dll is in the system dir, or in a directory in the PATH variable)
// Since we only tested with Python 3.9 and higher, we make this the minimum version. 
// Using 0x03020000 or even just 3 allows to work with lower Python versions, but is currently untested.
#define Py_LIMITED_API 0x03090000

// the following lines are a workaround for the problem 'cannot open file python39_d.lib'. This occurs because Python tries
// to link to the debug version of the library, even when that is not installed (and not really needed in our case).
// as a workaround, we trick the python.h include to think we are always building a Release build.
#ifdef _DEBUG
#undef _DEBUG
#pragma warning (disable: 4996)
#include <Python.h>
#define _DEBUG
#else
#pragma warning (disable: 4996)
#include <Python.h>
#endif

// see https://numpy.org/doc/stable/reference/c-api/array.html#c.import_array
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PyDevice_ARRAY_API

#define ERR_PYTHON_SCRIPT_NOT_FOUND 101
#define ERR_PYTHON_NO_DEVICE_DICT 102
#define ERR_PYTHON_DEVICE_NOT_FOUND 103
#define ERR_PYTHON_ONLY_ONE_HUB_ALLOWED 104
#define ERR_PYTHON_EXCEPTION 105
#define _check_(expression) if (auto result=(expression); result != DEVICE_OK) return result
