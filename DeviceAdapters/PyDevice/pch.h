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
#include "stable.h" // stable Python C API

/// Helper function for checking if a file exists
inline bool FileExists(const fs::path& path) noexcept {
    std::ifstream test(path);
    return test.good();
}

#define ERR_PYTHON_SCRIPT_NOT_FOUND 101
#define ERR_PYTHON_NO_DEVICE_DICT 102
#define ERR_PYTHON_DEVICE_NOT_FOUND 103
#define ERR_PYTHON_ONLY_ONE_HUB_ALLOWED 104
#define ERR_PYTHON_EXCEPTION 105
#define _check_(expression) if (auto result=(expression); result != DEVICE_OK) return result

#include <windows.h>

#pragma comment(lib, "python39")
#pragma comment(lib, "delayimp")
#pragma comment(lib, "user32")
