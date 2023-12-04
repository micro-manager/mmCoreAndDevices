## Build PyDevice yourself

# Install prerequisites
### Visual Studio with C++ build tools.
You can download the free Visual Studio Community edition here https://visualstudio.microsoft.com/free-developer-offers/. Make sure to include the c++ build tools during installation.


### Python 3
A list of required packages can be found in requirements.txt. Note that for the core functionality, only `numpy` needs to be installed. Make sure `python.exe` is on the search path

### Micro-Manager 2.0 executable
You can download the latest version (nightly build) here: https://micro-manager.org/Micro-Manager_Nightly_Builds. Alternatively, you can use an older, stable, version, with the caveat described below. We recommend installing in the default location: `C:\Program Files\Micro-Manager-2.0\`

### 3rd party public repository
This needs to have access to a Python install in order to build correctly. The required files are present in the 3rdpartypublic repository which hosts multiple build requirement files.
In theory this device could be built against any Python 3 install.


## When building all Micro-Manager devices:
This is the approach recommended by the Micro-Manager team

1. Open the solution file `mmCoreAndDevices/micromanager.sln` in Visual Studio
2. In the solution explorer, delete all projects, except for MMCore and MMDevice-SharedRuntime. (tip: select multiple projects using shift, and then press delete to remove them)
3. Right-click on the Solution in the Solution explorer, choose 'add->existing item' and browse to `mmCoreAndDevices\DeviceAdapters\PythonBinding\PythonBinding.vcxproj`
 If asked to convert to a newer version, press cancel.
5. Right-click on the Solution and choose 'Project dependencies...'. Make sure MMCore depends on MMDevice-SharedRuntime and PythonBinding depends on MMCore
6. Right-click on the PythonBinding project and 'set as startup project' 
7. Build the project. You will get an error (Python.h not found)
8. Build the project again, the error should have disappeared. If the build error python > autoconfig.props occurs, make sure your Python install is in PATH


# Troubleshooting
make sure you have the same version of micro-manager as you cloned from github. If the versions are different, the plugin will not be recognized.

For debugging, let Visual Studio start the Micromanager executable (ImageJ.exe). Make sure to enable Debug -> Other Debug Targets -> Child Process Debug Settings -> Enable Child Process Debugging.