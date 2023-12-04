## Build PyDevice yourself

# Install prerequisites
### Visual Studio with C++ build tools.
You can download the free Visual Studio Community edition here https://visualstudio.microsoft.com/free-developer-offers/. Make sure to include the c++ build tools during installation.


### Python 3
A list of required packages can be found in requirements.txt. Note that for the core functionality, only `numpy` needs to be installed. Make sure `python.exe` is on the search path

### Micro-Manager 2.0 executable
You can download the latest version (nightly build) here: https://micro-manager.org/Micro-Manager_Nightly_Builds. Alternatively, you can use an older, stable, version, with the caveat described below. We recommend installing in the default location: `C:\Program Files\Micro-Manager-2.0\`

### Micro-Manager SDK (mmCoreAndDevices)
Clone the `mmCoreAndDevices` repository, which contains the source code needed to interface with Micro-Manager. We recommend cloning the mmCoreAndDevices repo in one of the following locations:
* A sibling directory of the OpenWFS repository (e.g. if this repo is cloned here: `c:\git\openwfs`, put the SDK in `c:\git\mmCoreAndDevices`)
* Install OpenWFS as a subrepo of mmCoreAndDevices:

~~~
git clone https://github.com/micro-manager/mmCoreAndDevices
cd mmCoreAndDevices/DeviceAdapters
git submodule add https://github.com/ivovellekoop/pydevice
~~~

It is **essential** that the version of Micro-Manager and the mmCoreAndDevices match. Different versions have a different 'device-interface', which results in the plugin not being shown in the list in Micro-Manager. If you want to compile for an older (non nightly-build) version of Micro-Manager, you can checkout an old version of mmCoreAndDevices (check the git tags to find the correct version for a given device interface version)


# Configure Visual Studio
## Stand-alone build
When configuring as a stand-alone build (recommended), run the auto-configuration script
~~~
python autoconfig.py
~~~
If the script cannot find all folders, you can configure them manually later by editing the AutoConfig.props file.


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