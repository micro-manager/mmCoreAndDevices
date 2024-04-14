#include "pch.h"
#include "config.h"

fs::path pythonHome; // location of python3xx.dll
fs::path pythonVenv; // location of pyvenv.cfg
HMODULE pythonDll = nullptr; // handle to loaded python3xx.dll
PyThreadState* g_threadState = nullptr; // pointer to the thread state

const std::regex filenamePattern(R"(python3[0-9]+\.dll)", std::regex_constants::icase);
std::regex key_value(R"(home\s*=\s*(.+?)\s*$)");

PyObject* Py_None = nullptr;
PyObject* Py_True = nullptr;
PyObject* Py_False = nullptr;


/**
    @brief Tries to open and parse pyvenv.cfg

    When running in a virtual environment, we need to read the 'home' entry of the pyvenv.cfg configuration file,
    since it points to the location of the pythonxx.dll that we need to load.
    We have to load this dll before calling any Python API function, and we must load the correct one (there may be multiple python dlls,
    and we don't know the version number in advance).
*/
bool TryParsePyvenv(const fs::path& venv) noexcept
{
    auto configFile = std::ifstream(venv / "pyvenv.cfg");
    if (!configFile.is_open())
        return false; // file not found or could not be opened

    // locate the `home = ...` value
    string line;
    while (std::getline(configFile, line)) {
        std::smatch matches;
        if (std::regex_search(line, matches, key_value)) {
            pythonVenv = venv;
            pythonHome = matches.str(1);
            return true;
        }
    }
    return false; // pyvenv.cfg did not contain a 'home', we cannot use it
}

// Hook for the delay-loading the python dll. Instead of loading the dll, we return a handle to the pre-loaded dll
// which may be different from python39.dll (e.g. python311.dll)
FARPROC WINAPI delayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify == dliNotePreLoadLibrary && strcmp(pdli->szDll, "python39.dll") == 0) 
        return reinterpret_cast<FARPROC>(pythonDll);
    return nullptr;
}

ExternC const PfnDliHook __pfnDliNotifyHook2 = delayHook;


/**
 * \brief Loads the python3xx.dll runtime library using the specified folder to locate it
 *
 * If search is false, `venv` should be the root folder of a Python virtual environment. It should hold an `pyvenv.cfg` file
 * with a `home` variable that points to the folder holding python3xx.dll (i.e. PYTHONHOME).
 *
 * If search is true, and the folder does not hold `pyvenv.cfg`:
 * - recursively look in parent folders if they have a `venv` or `.venv` subfolder. If they do, check if they contain a `pyvenv.cfg` file and use that.
 * - if no `pyvenv.cfg` was found, use the $PYTHONHOME$ environment variable if present.
 * - otherwise, try to load `python3.dll` through the OS, and use the path of that folder as PYTHONHOME.
 *
 *
 * \param venv 
 * \param search 
 */
bool SetupPaths(const fs::path& venv, bool search)
{
    // option 1: load the specified virtual environment
    if (TryParsePyvenv(venv))
        return true;

    if (!search)
        return false;

    // option 2: look in parent folders for venv or .venv and load that virtual environment
    fs::path searchPath = venv;
    for (int depth = 0; depth < 10; depth++)
    {
        searchPath = searchPath.parent_path();
        if (searchPath.empty())
            break;

        if (TryParsePyvenv(searchPath / "venv") || TryParsePyvenv(searchPath / ".venv"))
            return true;
    }

    // option 3: check the PYTHONHOME environmental variable
    // in this case, there is no virtual environment
    if (auto pythonHomeVariable = _wgetenv(L"PYTHONHOME"))
    {
        pythonHome = pythonHomeVariable;
        return true;
    }

    // option 4: check if the OS can find python3.dll. If so, use that folder
    // in this case, there is no virtual environment
    if (auto handle = LoadLibrary(L"python3.dll"))
    {
        wchar_t dllPath[MAX_PATH];
        if (GetModuleFileName(handle, dllPath, MAX_PATH))
            pythonHome = fs::path(dllPath).parent_path();
        FreeLibrary(handle);
        return !pythonHome.empty();
    }

    return false;
}



/**
 * \brief Locates and loads the python3xx.dll runtime library
 * \param scriptPath
 */
bool InitializePython(const fs::path& venv, bool search) noexcept
{
    pythonHome.clear();
    pythonVenv.clear();
    if (pythonDll)
    {
        FreeLibrary(pythonDll);
        pythonDll = nullptr;
    }

    try {
        // Locate python home and python path
        if (!SetupPaths(venv, search))
            return false;

        // Search the home folder for a file of the name python3xx.dll
        // When found, load the dll
        for (const auto& entry : fs::directory_iterator(pythonHome))
            if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), filenamePattern))
            {
                pythonDll = LoadLibrary(entry.path().generic_wstring().c_str());
                if (!pythonDll)
                    return false; // could not load the python dll!

                // Load the data members from the DLL
                Py_None = reinterpret_cast<PyObject*>(GetProcAddress(pythonDll, "_Py_NoneStruct"));
                Py_True = reinterpret_cast<PyObject*>(GetProcAddress(pythonDll, "_Py_TrueStruct"));
                Py_False = reinterpret_cast<PyObject*>(GetProcAddress(pythonDll, "_Py_FalseStruct"));
                if (Py_None == nullptr || Py_False == nullptr || Py_True == nullptr) 
                    return false; // we loaded the dll, but it does not hold the none true and false data members!?
                break;
            }
        if (!pythonDll)
            return false; // there was no python dll in the home folder
    }
    catch (const fs::filesystem_error&)
    {
        return false; // there was an error accessing one of the files or iterating one of the folders
    }

    // Start the Python interpreter if it is not running yet
    // all paths are set up. When using the first function from the python dll (Py_IsInitialized below),
    // the delay-load mechanism will now load the correct dll
    if (!Py_IsInitialized()) {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        config.site_import = 1;

        // after a lot of trial and error, the method below seems to work
        // it makes the Python runtime open the pyvenv.cfg file and parse it,
        // and add the pythonVenv / "Lib" / "site-packages" to the system path
        if (!pythonVenv.empty())
        {
            config.isolated = 1;
            auto program = (pythonVenv / "Scripts" / "python.exe").make_preferred();
            PyConfig_SetString(&config, &config.program_name, program.c_str());
            PyConfig_SetString(&config, &config.executable, program.c_str());
        } else
        {
            // not using a virtual environment. Let Python figure out the paths itself.
            config.isolated = 0;
        }
        auto status = Py_InitializeFromConfig(&config);
        PyConfig_Read(&config);
        if (PyStatus_Exception(status))
            return false;

        // enable multi-threading
        if (!g_threadState)
            g_threadState = PyEval_SaveThread();
    }

    return true;
}

