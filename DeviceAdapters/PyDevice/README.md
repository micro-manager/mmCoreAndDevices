# PyDevice
Adapter to import Python scripts as devices in micro-manager

# Installation
1. Prerequisites
   * Visual Studio with C++ build tools.
   * Python
   * Micro-manager (nightly build)

2. If you don't have the micro-manager source installed yet, clone micro-manager and its submodules from github
~~~
git clone https://github.com/micro-manager/micro-manager
cd micro-manager
git submodule update --init --recursive 
~~~

3. Add the Python binding submodule
~~~
cd mmCoreAndDevices/DeviceAdapters
git submodule add https://github.com/ivovellekoop/pydevice PyDevice
~~~

4. Set up the project in Visual studio
    1. Open the solution file 'micro-manager/mmCoreAndDevices' in Visual Studio
    2. In the solution explorer, delete all projects, except for MMCore and MMDevice-SharedRuntime. (tip: select multiple projects using shift, and then press delete to remove them)
    3. Right-click on the Solution in the Solution explorer, choose 'add->existing item' and browse to micromanager-bindings\micro-manager\mmCoreAndDevices\DeviceAdapters\PythonBinding\PythonBinding.vcxproj
 If asked to convert to a newer version, press cancel.
    5. Right-click on the Solution and choose 'Project dependencies...'. Make sure MMCore depends on MMDevice-SharedRuntime and PythonBinding depends on MMCore
    6. Right-click on the PythonBinding project and 'set as startup project' 
    7. Build the project. You will get an error (Python.h not found)
    8. Build the project again, the error should have disappeared. If the build error python > autoconfig.props occurs, make sure your Python install is in PATH


5. Note: make sure you have the same version of micro-manager as you cloned from github. If the versions are different, the plugin will not be recognized.

# Troubleshooting
....
