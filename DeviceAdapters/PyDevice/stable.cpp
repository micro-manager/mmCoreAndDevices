#include "pch.h"
#include "config.h"

fs::path pythonHome; // location of python3xx.dll
fs::path pythonPath; // location of site packages
fs::path pythonDll;  // location of python3xx.dll + actual file name
fs::path pythonVenv; // location of pyvenv.cfg
const std::regex filenamePattern(R"(python3[0-9]+\.dll)", std::regex_constants::icase);
std::regex key_value(R"(\s*(.+?)\s*=\s*(.+?)\s*$)");

PyObject* Py_None = nullptr;
PyObject* Py_True = nullptr;
PyObject* Py_False = nullptr;


/**
    @brief Tries to open and parse pyvenv.cfg
*/
bool TryParsePyvenv(const fs::path& venv) noexcept
{
    auto configFile = std::ifstream(venv / "pyvenv.cfg");
    if (!configFile.is_open())
        return false; // file not found or could not be opened

    // parse line by line
    string line;
    while (std::getline(configFile, line)) {
        std::smatch matches;
        if (std::regex_search(line, matches, key_value)) {
            if (matches.str(1) == "home")
                pythonHome = matches.str(2);
            //            else if (matches[1] == "include-system-site-packages")
        }
    }
    if (!pythonHome.empty())
    {
        pythonPath = venv / "Lib" / "site-packages";
        pythonVenv = venv;
        return true;
    }
    return false;
}

// TODO: load the library outside of the hook to catch errors without crashing the plugin
FARPROC WINAPI delayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify == dliNotePreLoadLibrary) {
        if (strcmp(pdli->szDll, "python39.dll") == 0) { // Check the DLL name
            // Specify the path to your DLL
            HMODULE hModule = LoadLibrary(pythonDll.generic_wstring().c_str());

            // Load the data members from the DLL
            Py_None = reinterpret_cast<PyObject*>(GetProcAddress(hModule, "_Py_NoneStruct"));
            Py_True = reinterpret_cast<PyObject*>(GetProcAddress(hModule, "_Py_TrueStruct"));
            Py_False = reinterpret_cast<PyObject*>(GetProcAddress(hModule, "_Py_FalseStruct"));
            if (Py_None == nullptr || Py_False == nullptr || Py_True == nullptr) {
                return nullptr;
            }

            return reinterpret_cast<FARPROC>(hModule);
        }
    }
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

        if (TryParsePyvenv(searchPath / "venv"))
            return true;
        if (TryParsePyvenv(searchPath / ".venv"))
            return true;
    }

    // option 3: check the PYTHONHOME environmental variable
    auto pythonHomeVariable = _wgetenv(L"PYTHONHOME");
    pythonHome = pythonHomeVariable ? pythonHomeVariable : L"";
    if (!pythonHome.empty())
    {
        auto pythonPathVariable = _wgetenv(L"PYTHONPATH");
        pythonPath = pythonPathVariable ? pythonPathVariable : L"";
        return true;
    }

    // option 4: check if the OS can find python3.dll. If so, use that folder
    if (auto handle = LoadLibrary(L"python3.dll"))
    {
        wchar_t dllPath[MAX_PATH];
        if (GetModuleFileName(handle, dllPath, MAX_PATH))
        {
            pythonHome = fs::path(dllPath).parent_path();
            pythonPath.clear();
        }
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
    pythonPath.clear();
    pythonHome.clear();
    pythonDll.clear();

    try {
        // Locate python home and python path
        if (!SetupPaths(venv, search))
            return false;

        // Search the home folder for a file of the name python3xx.dll
        for (const auto& entry : fs::directory_iterator(pythonHome))
            if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), filenamePattern))
            {
                pythonDll = entry;
                break;
            }
        if (pythonDll.empty())
            return false;

    }
    catch (const fs::filesystem_error&)
    {
        return false;
    }

    // Start the Python interpreter if it is not running yet
    // all paths are set up. When using the first function from the python dll (Py_IsInitialized below),
    // the delay-load mechanism will now load the correct dll
    if (!Py_IsInitialized()) {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        config.isolated = 1;
        config.site_import = 1;

        // after a lot of trial and error, the method below seems to work
        // it makes the Python runtime open the pyvenv.cfg file and parse it,
        // and add the pythonVenv / "Lib" / "site-packages" to the system path

        auto program = (pythonVenv / "Scripts" / "python.exe").make_preferred();
        PyConfig_SetString(&config, &config.program_name, program.c_str());
        PyConfig_SetString(&config, &config.executable, program.c_str());
        auto status = Py_InitializeFromConfig(&config);
        PyConfig_Read(&config);
        if (PyStatus_Exception(status))
            return false;
    }
    return true;
}