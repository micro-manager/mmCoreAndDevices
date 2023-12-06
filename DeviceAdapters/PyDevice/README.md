# PyDevice
PyDevice is a MicroManager device adapter that loads Python scripts as MicroManager devices.

## How does it work?
 - The PyDevice runs a Python interpreter in which the Python functions are executed. 
   Due to following specific naming conventions, the properties and methods of objects can be translated to MicroManager properties and methods

# Known limitations
* PyDevice was developed and tested on Windows. If you are interested in porting the plugin to Linux, please contact the developers. 

# Build instructions
## Prerequisites
First, install the following prerequisites
* **Visual Studio with C++ build tools**.
You can download the free Visual Studio Community edition here https://visualstudio.microsoft.com/free-developer-offers/. Make sure to include the c++ build tools during installation.
* **Python 3**.
To build PyDevice, Python 3.9 or later and the `numpy` package need to be installed. The compiler looks for the Python installation in a folder `3rdpartypublic/Python`, where `3rdpartypublic` is located in the parent folder of the parent folder of `mmCoreAndDevices`. For example, if the repository is located in `c:\git\mmCoreAndDevices`, we expect to find the include file at `c:\3rdpartypublic\Python\include\Python.h`. Instead of installing Python in that folder directly, it is recommended to make a symbolic link to the Python install. First, open a terminal window with administrator privileges, and navigate to the `3rdpartypublic` directory. Then create the symbolic link to the Python install, e.g. `mklink /D Python C:\Users\{username}\anaconda3`.

* **Micro-Manager 2.0**
You can download the latest version (nightly build) here: https://micro-manager.org/Micro-Manager_Nightly_Builds. Alternatively, you can build the micro-manager application from source, or use an older, stable, version. Note that Micro-Manager only recognizes plugins with the correct internal version number, so if the Mirco-Manager version is too old, it will not recognize the plugin.


## Building the plugin
1. Open the solution file `mmCoreAndDevices/micromanager.sln` in Visual Studio
 If asked to convert to a newer version, press cancel.
2. Not all plugins will build correctly. To build just the PyDevice plugin, right-click the PyDevice in the Solution Explorer and select `build`.

## Debugging
To debug the plugin in Micro-Manager, select a `Debug` build configuration in Visual Studio. In addition, right-click PyDevice in the solution explorer, and `Set as Startup Project`. Right-click again and edit the PyDevice project Properties, under Debugging, fill in the following settings:

| | |
|---------|--------------------------------|
| Command | $(ProjectDir)/debug_helper.bat |
| Command Arguments | $(TargetPath) |
| Environment |MM_EXECUTABLE_PATH="C:\Program Files\Micro-Manager-2.0"|

Here, adjust the `MM_EXECUTABLE_PATH` if you installed Micro-Manager somewherre else.
Finally, in the menu select `Debug->Other Debug Targets->Child Process Debug Settings...` and tick `Enable child process debugging`.
You can now press F5 to build and install the plugin, and start Micro-Manager with the debugger attached. Note: you can safely ignore the segmentation faults occuring in the java virtual machine.


# Troubleshooting
make sure you have the same version of micro-manager as you cloned from github. If the versions are different, the plugin will not be recognized.


## How do I use it?
 - If the device is available in your MicroManager install, you can test it by loading the test.py file. 
   When adding the PyHub in the config. manager, the boxes 'ModulePath' and 'ScriptPath' appear. ModulePath signifies the location of your Python  
   home. This can be a virtual environment, an Anaconda install or just a Python install, as long as its Python 3.
 
   If you leave the ModulePath on '(auto)' it will most likely find a Python install for you. If the script you load is in a virtual environment,
   It will find the virtual environment for you. This is important as it contains specific packages (like NumPy) that your scripts need to run.

## How do I build my own devices?
 - We have clarified that in more detail in the subfolder Python_devices,  starting with HOW_TO_MAKE_A_DEVICE.md. 