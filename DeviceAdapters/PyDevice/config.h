#pragma once

struct PyStatus{
    enum {
        _PyStatus_TYPE_OK = 0,
        _PyStatus_TYPE_ERROR = 1,
        _PyStatus_TYPE_EXIT = 2
    } _type;
    const char* func;
    const char* err_msg;
    int exitcode;
};

struct PyWideStringList{
    /* If length is greater than zero, items must be non-NULL
       and all items strings must be non-NULL */
    Py_ssize_t length;
    wchar_t** items;
};


struct PyPreConfig {
    int _config_init;     /* _PyConfigInitEnum value */
    int parse_argv;
    int isolated;
    int use_environment;
    int configure_locale;
    int coerce_c_locale;
    int coerce_c_locale_warn;

//#ifdef MS_WINDOWS
    int legacy_windows_fs_encoding;
//#endif
    int utf8_mode;
    int dev_mode;
    int allocator;
};

struct PyConfig {
    int _config_init;
    int isolated;
    int use_environment;
    int dev_mode;
    int install_signal_handlers;
    int use_hash_seed;
    unsigned long hash_seed;
    int faulthandler;
    int _use_peg_parser;
    int tracemalloc;
    int import_time;
    int show_ref_count;
    int dump_refs;
    int malloc_stats;
    wchar_t* filesystem_encoding;
    wchar_t* filesystem_errors;
    wchar_t* pycache_prefix;
    int parse_argv;
    PyWideStringList argv;
    wchar_t* program_name;
    PyWideStringList xoptions;
    PyWideStringList warnoptions;
    int site_import;
    int bytes_warning;
    int inspect;
    int interactive;
    int optimization_level;
    int parser_debug;
    int write_bytecode;
    int verbose;
    int quiet;
    int user_site_directory;
    int configure_c_stdio;
    int buffered_stdio;
    wchar_t* stdio_encoding;
    wchar_t* stdio_errors;
//#ifdef MS_WINDOWS
    int legacy_windows_stdio;
//#endif
    wchar_t* check_hash_pycs_mode;
    int pathconfig_warnings;
    wchar_t* pythonpath_env;
    wchar_t* home;
    int module_search_paths_set;
    PyWideStringList module_search_paths;
    wchar_t* executable;
    wchar_t* base_executable;
    wchar_t* prefix;
    wchar_t* base_prefix;
    wchar_t* exec_prefix;
    wchar_t* base_exec_prefix;
    int skip_source_first_line;
    wchar_t* run_command;
    wchar_t* run_module;
    wchar_t* run_filename;
    int _install_importlib;
    int _init_main;
    int _isolated_interpreter;
};
/*
struct PyConfig {
    int _config_init;

    int isolated;
    int use_environment;
    int dev_mode;
    int install_signal_handlers;
    int use_hash_seed;
    unsigned long hash_seed;
    int faulthandler;
    int tracemalloc;
    int perf_profiling;
    int import_time;
    int code_debug_ranges;
    int show_ref_count;
    int dump_refs;
    wchar_t* dump_refs_file;
    int malloc_stats;
    wchar_t* filesystem_encoding;
    wchar_t* filesystem_errors;
    wchar_t* pycache_prefix;
    int parse_argv;
    PyWideStringList orig_argv;
    PyWideStringList argv;
    PyWideStringList xoptions;
    PyWideStringList warnoptions;
    int site_import;
    int bytes_warning;
    int warn_default_encoding;
    int inspect;
    int interactive;
    int optimization_level;
    int parser_debug;
    int write_bytecode;
    int verbose;
    int quiet;
    int user_site_directory;
    int configure_c_stdio;
    int buffered_stdio;
    wchar_t* stdio_encoding;
    wchar_t* stdio_errors;
#ifdef MS_WINDOWS
    int legacy_windows_stdio;
#endif
    wchar_t* check_hash_pycs_mode;
    int use_frozen_modules;
    int safe_path;
    int int_max_str_digits;

    int cpu_count;
#ifdef Py_GIL_DISABLED
    int enable_gil;
#endif
    int pathconfig_warnings;
    wchar_t* program_name;
    wchar_t* pythonpath_env;
    wchar_t* home;
    wchar_t* platlibdir;
    int module_search_paths_set;
    PyWideStringList module_search_paths;
    wchar_t* stdlib_dir;
    wchar_t* executable;
    wchar_t* base_executable;
    wchar_t* prefix;
    wchar_t* base_prefix;
    wchar_t* exec_prefix;
    wchar_t* base_exec_prefix;
    int skip_source_first_line;
    wchar_t* run_command;
    wchar_t* run_module;
    wchar_t* run_filename;
    wchar_t* sys_path_0;
    int _install_importlib;
    int _init_main;
    int _is_python_build;

#ifdef Py_STATS
    // If non-zero, turns on statistics gathering.
    int _pystats;
#endif

#ifdef Py_DEBUG
    // If not empty, import a non-__main__ module before site.py is executed.
    // PYTHON_PRESITE=package.module or -X presite=package.module
    wchar_t* run_presite;
#endif
};*/



// These functions are _not_ part of the stable API, but they are the only more-or-less reliable way to initialize Python
// The old-fashioned way (using Py_Initialize) _is_ part of the stable API, but it is deprecated!
// Also, it crashes the program if Python cannot be loaded.
extern "C" {
    __declspec(dllimport) int Py_SetStandardStreamEncoding(const char* encoding, const char* errors);
    __declspec(dllimport) PyStatus Py_PreInitialize(const PyPreConfig* src_config);
    __declspec(dllimport) PyStatus Py_PreInitializeFromBytesArgs(const PyPreConfig* src_config, Py_ssize_t argc, char** argv);
    __declspec(dllimport) PyStatus Py_PreInitializeFromArgs(const PyPreConfig* src_config, Py_ssize_t argc, wchar_t** argv);
    __declspec(dllimport) PyStatus Py_InitializeFromConfig(const PyConfig* config);
    __declspec(dllimport) PyStatus _Py_InitializeMain();
    __declspec(dllimport) int Py_RunMain(void);
    __declspec(dllimport) int Py_FdIsInteractive(FILE*, const char*);

    __declspec(dllimport) PyStatus PyStatus_Ok(void);
    __declspec(dllimport) PyStatus PyStatus_Error(const char* err_msg);
    __declspec(dllimport) PyStatus PyStatus_NoMemory(void);
    __declspec(dllimport) PyStatus PyStatus_Exit(int exitcode);
    __declspec(dllimport) int PyStatus_IsError(PyStatus err);
    __declspec(dllimport) int PyStatus_IsExit(PyStatus err);
    __declspec(dllimport) int PyStatus_Exception(PyStatus err);
    __declspec(dllimport) PyStatus PyWideStringList_Append(PyWideStringList* list, const wchar_t* item);
    __declspec(dllimport) PyStatus PyWideStringList_Insert(PyWideStringList* list, Py_ssize_t index, const wchar_t* item);
    __declspec(dllimport) void PyPreConfig_InitPythonConfig(PyPreConfig* config);
    __declspec(dllimport) void PyPreConfig_InitIsolatedConfig(PyPreConfig* config);
    __declspec(dllimport) void PyConfig_InitPythonConfig(PyConfig* config);
    __declspec(dllimport) void PyConfig_InitIsolatedConfig(PyConfig* config);
    __declspec(dllimport) void PyConfig_Clear(PyConfig*);
    __declspec(dllimport) PyStatus PyConfig_SetString(PyConfig* config, wchar_t** config_str, const wchar_t* str);
    __declspec(dllimport) PyStatus PyConfig_SetBytesString(PyConfig* config, wchar_t** config_str, const char* str);
    __declspec(dllimport) PyStatus PyConfig_Read(PyConfig* config);
    __declspec(dllimport) PyStatus PyConfig_SetBytesArgv(PyConfig* config, Py_ssize_t argc, char* const* argv);
    __declspec(dllimport) PyStatus PyConfig_SetArgv(PyConfig* config, Py_ssize_t argc, wchar_t* const* argv);
    __declspec(dllimport) PyStatus PyConfig_SetWideStringList(PyConfig* config, PyWideStringList* list, Py_ssize_t length, wchar_t** items);
}