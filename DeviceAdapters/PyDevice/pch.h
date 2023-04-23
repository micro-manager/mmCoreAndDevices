#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <functional>
#include <filesystem>
#pragma warning(disable: 5040) // disable warning we get because we are using C++17 for compilation.
#include <DeviceBase.h> // all MM includes

// the following lines are a workaround for the problem 'cannot open file python39_d.lib'. This occurs because Python tries
// to link to the debug version of the library, even when that is not installed (and not really needed in our case).
// as a workaround, we trick the python.h include to think we are always building a Release build.
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#define _DEBUG
#else
#include <Python.h> // if you get a compiler error here, try building again and see if magic happens
#endif

// see https://numpy.org/doc/stable/reference/c-api/array.html#c.import_array
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PyDevice_ARRAY_API
