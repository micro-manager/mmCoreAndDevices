
| | |
|:------------|:--------------------------------------------------------------------|
| **Summary** | Device adapter for the integration of Python objects in MicroManager |
| **Author**  | Ivo Vellekoop and Jeroen Doornbos                                    |
| **License** | BSD-3                                                               |
| **Devices** | Stage, XYStage, Camera, Generic device                               |


# PyDevice
PyDevice is a Micro-Manager device adapter that imports objects from a Python script, and integrates them into MicroManager as devices (e.g. a camera or a stage). This integration enables the use of Python scripts to control microscope hardware, without requiring any programming experience from the end user. Essentially, PyDevice acts as a translator, allowing Python-developed objects to be used in MicroManager with almost no interfacing code required.

PyDevice is currently **experimental**. It may still change significantly, and it is not fully tested under all circumstances. Please feel free to try it out and provide feedback.

## Requirements
To use PyDevice, the following software needs to be installed:

* **Windows 10 or 11**. PyDevice was developed and tested in Windows. Although it _should_ work on Linux, this is not tested. Please contact the developers if you are interested in testing and using PyDevice in Linux.

* **Python 3.9 or higher**. 

* **Micro-Manager 2.0** with the PyDevice device adapter installed. Until PyDevice is included in the nightly build (https://micro-manager.org/Micro-Manager_Nightly_Builds), it is still required to build the device adapter from source, as described below.

## Getting started
1. PyDevice needs to be able to locate the Python runtime. It is recommended to set up a virtual environment for your project, as described in [Virtual environments](#virtual-environments). Alternatively, if PyDevice does not find a virtual environment, it will try to locate the global Python installation, which should be included in the system path. To verify that a global Python installation is configured correctly, type `python --version` in a command prompt.

2. Create a script that defines a class with properties and methods that you want to use in Micro-Manager. For example, create a file `hello_device.py` with the following content:

```python
class HelloDevice:
    def __init__(self):
        self._message = "Hello, World!"

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, value):
        self._message = str(value)


devices = {'hello': HelloDevice()}
```

3. In Micro-Manager, create a new hardware configuration. In the Hardware Configuration Wizard select `PyDevice->PyHub` from the list of devices and `Add` the device. 

4. There is no need to set the `PythonEnvironment` or `ScriptPath` properties. Instead, just press `Ok` to get a file browser dialog. Select the `hello_device.py` script you just created. This will execute the script and import all objects in the `devices` dictionary into Micro-Manager. You will now see a list of the objects that were successfully recognized, similar to the following screen:

![camera_selection_screen](docs/camera_selection_screen.png)

5. Select the device to add it to the configuration. Optionally, change the label from `Device:hello` to anything you like, and press `Ok` to close the dialog.

6. Complete the rest of the wizard, add other devices if you want. Note, however, that there can only be a single PyHub device, and thus a single Python script, in a configuration. This script, however, can define as many devices as needed by adding them to the `devices` dictionary.

If all went well, you should now see the device in the Device Property Browser, with one property `Message`, that you can modify.

## How does it work?

The PyDevice device adapter runs a Python interpreter in which the Python script is executed. It then looks for a dictionary called `devices`, and adds the objects in this dictionary as devices to Micro-Manager.

To make a property available in Micro-Manager, it should be public (i. e., not `_`-prefixed), and declared with the `@property` decorator, as shown in the example. Moreover, the property getter should have a type annotation for the return value (`-> str`), so that PyDevice can determine the type of the property. PyDevice supports `str`, `int`, `float`, `bool`, as well as `Enum` types and floats with units (see [Advanced use](#advanced-use)). If you want to make the property writable, you should also define a setter method (`@message.setter` in the example). 

Note that the property name in the example was converted from `message` to `Message` in Micro-Manager to comply with the naming conventions in both Python and Micro-Manager. Also note that, except for the construction of the dictionary object, there is no special code to interact with Micro-Manager. The Python script can be used and tested independently of Micro-Manager.


## Other device types
It is also possible to define other device types. PyDevice will automatically detect the device type by examining the properties and methods present on the object. Currently, the following device types are supported:

- `Camera`: requires the following properties and methods:
    - `exposure_ms` (float): the exposure time in milliseconds
    - `top` (int): the top coordinate of the region of interest
    - `left` (int): the left coordinate of the region of interest
    - `width` (int): the width of the region of interest
    - `height` (int): the height of the region of interest
    - `binning` (int): the binning factor. This property is optional, and defaults to 1
    - `read()` (method): acquire an image and return it as a numpy array, or as any object that implements the Python buffer protocol (such as a pytoch object).
    - `busy()` (method): return `True` if the camera is busy acquiring an image

- `Stage`: requires the following properties and methods:
    - `position_um` (float): position of the stage in micrometer
    - `step_size_um` (float): step size of the stage in micrometer
    - `home()` (method): move the stage to the home position and reset the position to (0)
    - `busy()` (method): return `True` if the stage is moving

- `XYStage`: requires the following properties and methods:
    - `x_um` (float): position of the stage in micrometer
    - `y_um` (float): position of the stage in micrometer
    - `step_size_x_um` (float): step size of the stage in micrometer
    - `step_size_y_um` (float): step size of the stage in micrometer
    - `home()` (method): move the stage to the home position and reset the position to (0, 0)
    - `busy()` (method): return `True` if the stage is moving

For examples of how to define devices, see the `examples` directory in the GitHub repository.

## Virtual environments
It is considered good practice to use virtual environments to manage Python packages. A virtual environment acts as a stand-alone Python installation, with its own packages and dependencies. This way, you can have different versions of packages for different projects, without interfering with each other. There are several different tools to set up a virtual environment, such as [`venv`](https://docs.python.org/3/tutorial/venv.html), `poetry`, and `conda`. 

When creating a PyDevice object, the `PythonEnvironment` variable may be used to specify the path to the virtual environment that should be used. It should point to a folder that has a `pyvenv.cfg` file, which is the configuration file that is part of a Python virtual environment. For example, Poetry stores the virtual environments in `C:\Users\{username}\AppData\Local\pypoetry\Cache\virtualenvs\{virtual-environment-name}\Lib\site-packages`.

To facilitate the use of virtual environments, the `PythonEnvironment` property is set to `(auto)` by default. In this case, PyDevice will look in the parent folders of the loaded script for a `venv` or `.venv` directory. If this directory is found and contains a `pyvenv.cfg` file, this virtual environment is used for running the Python code in the device script. 

If no virtual environment is found, the base Python installation of the system is used. To locate this install, first the `PYTHONHOME` environment variable is checked, and if that is not set, the system path is searched for the `python3.dll` file. Although using the system-wide Python installation can be convenient, it may lead to conflicts with other packages that are installed in the base Python installation. Therefore, it is recommended to use a virtual environment.

## Use with `openwfs`
PyDevice was developed to integrate seamlessly with our [OpenWFS](https://openwfs.readthedocs.io) package for controlling wavefront shaping experiments. This package provides a high-level interface to control spatial light modulators, cameras, and other hardware. For example, it includes code to control a **_scanning multiphoton microscope_** through a NI-daq data acquisition card. This code integrates directly as a `Camera` device in Micro-Manager. In addition, it includes code to **_simulate a motorized microscope_**, which allows you to try out Micro-Manager and PyDevice without any hardware. The code for these devices is available in the `openwfs` package, which can be installed with `pip install openwfs`.


## Use in `pymmcore`
PyDevice can be used in [pymmcore](https://github.com/micro-manager/pymmcore) just as any other device adapter. For example, to load a configuration file `camera.cfg` that contains the definition of a camera device, and acquire an image, you can use the following code:

```python
import pymmcore
mmc = pymmcore.CMMCore()
mmc.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0/"])
mmc.loadSystemConfiguration("camera.cfg")
mmc.setProperty("cam", "Width", 121)
mmc.setProperty("cam", "Height", 333)
mmc.snapImage()
frame = mmc.getImage()
```

Using `pymmcore`, you can even debug the Python code, by simply setting a breakpoint in the code implementing the Device (tested in PyCharm 2024 with Conda 22.9 (Python 3.9)). This is actually quite remarkable as the device is running in a separate Python interpreter, and the debugger is attached to the main Python interpreter. 


## Advanced use
### astropy.units
Some of the properties that PyDevice expects are specified in milliseconds or micrometers (indicated by the `_ms` or `_um` suffix). As an alternative to using plain floats, you can use the `astropy.units` package to specify the units of the property. For example, the `exposure_ms` property in the `Camera` device can be defined as follows:

```python
from astropy import units as u

class Camera:
    def __init__(self):
        self._exposure = 10 * u.ms
        
    @property
    def exposure(self) -> u.Quantity[u.ms]:
        return self._exposure * u.ms

    @exposure.setter
    def exposure(self, value):
        self._exposure = value.to(u.ms)

    ...
```
PyDevice detects properties of type `astropy.units.Quantity` and automatically postfixes the unit in Micro-Manager (e. g., `Exposure-ms`). Currently, the following units are recognized: s, ms, us, ns, m, cm, mm, um, nm, A, mA, uA, V, mV, uV, Hz, kHz, MHz, GHz. One of the benefit of using this approach is that the user can specify the exposure time in the units they prefer (e.g. `camera.exposure = 1 * u.s`), with astropy taking care of the unit conversion.

### Enum types
PyDevice supports properties with Enum types. For example, the following code defines a property `color` with an Enum type:

```python
from enum import Enum
class Color(Enum):
    Orange = 1
    Red = 2
    Blue = 3
    
class DeviceWithColor:

    def __init__(self, color):
        self._color = color

    @property
    def color(self) -> Color:
        return self._color

    @color.setter
    def options(self, value: Color):
        self._color = Color(value)

    ...
```
PyDevice will detect that the `color` property returns an Enum, and create a property `Color` in Micro-Manager with the options `Orange`, `Red`, and `Blue` shown in a drop-down box as shown below

  ![example device full](docs/example_device_full.png)   


### Actions
`Camera` objects define a method to read a frame, and `Stage` and `XYStage` devices define methods to home the stage. These actions are started by Micro-Manager. Currently, there is no direct support for user-defined actions that can be started from the GUI. As a work-around, one can define a property that triggers an action when set. For example, the following code defines a property `capture` that triggers the `read` method of the `Camera` object:

```python
from enum import Enum
class Action(Enum):
    Idle = 0
    Do_A = 1
    Do_B = 2
    Do_C = 3
    
class DeviceWithActions:
    @property
    def action(self) -> Action:
        return Action.Idle

    @action.setter
    def options(self, value):
        if value == Action.Do_A:
            self.do_A()
        elif value == Action.Do_B:
            self.do_B()
        elif value == Action.Do_C:
            self.do_C()

    ...
```
The user can now select an action from a drop-down box in Micro-Manager to activate the device.



### Specifying ranges (experimental)
PyDevice supports specifying a range for integer and float properties using the `Annotated` type hint and the `Ge`, `Le` and `Interval` classes defined in the `annotated_types` package. For example, the type hint
  `Annotated[int, Ge(1), Le(42)}]` specifies that the property holds an integer between 1 and 42 inclusive. If both upper and lower limits are set, the Micro-Manager GUI displays a slider that the user can adjust the value with.

```python
from typing import Annotated
from annotated_types import Interval

class DeviceWithInteger:
    def __init__(self):
        self._integer = 0

    @property
    def integer(self) -> Annotated[int, Interval(ge=0,le=0)]:
        return self._integer
    
    @integer.setter
    def integer(self, value):
        self._integer = int(value)
```

### Using attributes instead of properties (experimental)
It is now possible to use attributes instead of properties. This can be useful when the property does not require any additional logic in the getter or setter. For example, the following code defines a device with an attribute `value`:

```python
class DeviceDirect:
    value: float
    def __init__(self):
        self.value = 0.0
```
Just as when using properties, the attribute should be public and have an appropriate type hint.    
    
## Known limitations
* PyDevice was developed and tested on Windows. If you are interested in porting the plugin to Linux, please contact the developers.
* It is not yet possible to link an action to a push button in the GUI. 
* Only a single PyHub device can be active at a time. If you want to combine multiple Python devices, just create a single Python scripts that collects all devices in a single `devices` dictionary.
* Inheriting property definitions from a base class may work, but this aspect is not fully tested yet.

## Troubleshooting
* **The plugin does not show up in Micro-Manager.** This may be because you have a version of Micro-Manager that does not have PyDevice included yet.

* **The PyDevice plugin is shown in the list of device adapters, but it grayed out and cannot be loaded**. If you built PyDevice yourself, this problem can be caused by a version difference between MicroManager executable and the source code used to build PyDevice.

* **The plugin crashes when it tries to load a script.** To find out what is the exact problem, enable logging in Micro-Manager and examine the core log file. The most likely cause is a problem in initializing the Python runtime. Unfortunately, in the case of an initialization problem, the Python runtime terminates the program instead of reporting an error. One of the causes may be an incorrect configuration of the system path or the `PYTHONHOME` environment variable. Try setting up a virtual environment as described above. 

* **My device script is recognized as `Device` instead of a `Camera`, `Stage`, etc.**. PyDevice determines the device type by examining the properties and methods of the object. If the object has all the required properties and methods of a `Camera`, it will be recognized as such. If the object is recognized as a `Device`, it means that one or more of the required properties or methods are missing. Check the spelling of the property names, and make sure that the property getters have the correct type annotation for the return value. For troubleshooting, you can construct a `PyDevice` wrapper object in Python, using the definition in `bootstrap.py`, which is included in the GitHub repository. See `tests_reflection.py` for an example. 

* **A property does not show up in MicroManager**. This is typically caused by a missing type annotation. Also note that properties should have an explicit 'getter', in the form of a `@property` decorator, see examples above. 

* **An error occurs when reading a property set to None**. Currently, MicroManager does not support missing values for properties. Instead, if `None` is found in a float property, it is silently converted to `nan`. For other property types, if the getter returns `None`, an error is given.

* * **<TypeError>**. Python does not enforce correct data types. This means that a property that is declaredw with a `-> int` type hint, may just as well return a `float`, `None`, or any other object. PyDevice will try to convert the returned value to the expected type, but if this fails, a `TypeError` is raised. To prevent this, make sure that the property holds a value of the correct type, which can be done by explictly converting a value to the correct type using the property setters, (e.g. `self._value = int(value)`, see examples above).




## Building from source code
First, install the following prerequisites

* **Visual Studio with C++ build tools**.
  You can download the free Visual Studio Community edition
  here <https://visualstudio.microsoft.com/free-developer-offers/>. Make sure to include the c++ build tools during
  installation.
* **Python 3**.
  To build PyDevice, the file `python39.lib` is needed. This file should be located in the folder `3rdpartypublic/Python/cp39-win_amd64libs/libs`, where `3rdpartypublic` is located in the parent folder of the parent folder of `mmCoreAndDevices`. For example, if the repository is located in `c:\git\mmCoreAndDevices`, we expect to find the file at `c:\3rdpartypublic\Python\cp39-win_amd64libs\libs\python39.lib`. This file can be copied from the installation directory of a Python 3.9 installation (e. g. <https://www.python.org/ftp/python/3.9.1/>).

* **Micro-Manager 2.0**
  You can download the latest version (nightly build) here: https://micro-manager.org/Micro-Manager_Nightly_Builds. Alternatively, you can build the micro-manager application from source, or use an older, stable, version. Note that Micro-Manager only recognizes plugins with the correct internal version number, so if the Mirco-Manager version is too old, it will not recognize the plugin.

### Building the plugin

1. Open the solution file `mmCoreAndDevices/Micro-Manager.sln` in Visual Studio
   If asked to convert to a newer version, press cancel.
2. Not all plugins will build correctly. To build just the PyDevice plugin, right-click the PyDevice in the Solution
   Explorer and select `build`.

### Implementation of loading the Python runtime
By far the hardest part of developing PyDevice was starting the Python runtime. Starting a virtual environment from c++ code is not trivial, and the process is poorly documented and extremely complex (a partial documentation of how the Python runtime locates dependencies can be found [here](https://github.com/python/cpython/blob/main/Modules/getpath.py)). The complication is, in part, caused by the following:

* c++ code that interfaces with Python should load the Python runtime. The _standard_ way to do this is to link the code with an import library, which causes the Python runtime to be loaded when the c++ code is loaded. The caveats here are: 
  1) the exact version of Python needs to be known at compile time. This means that if the plugin is built using Python 3.10, it will not work with Python 3.9, or Python 3.10 or any other Python version on the system of the user.
  2) The OS should be able to locate the runtime dll for exactly that version, or it should be distributed along with the device adapter.
  3) If the Python runtime cannot be found, the plugin fails to load altogether, without any error message, making it hard for the user to find out what is going on.

* In an attempt to mitigate these issues, Python defines a 'stable' API that allows linking against a generic `python3.dll` file, which then loads the runtime that is available on the system. This approach, however, has many problems on its own:
  1) With this approach, the plugin always loads the Python version that is present on the system path. There is no way to configure it to use a different Python version. So, a global Python installation must be available and there is no option to choose a different Python version (e.g. when using a virtual environment).
  2) The 'stable' API is not very stable. In particular, the way the Python runtime is initialized is now deprecated, with no stable alternative provided. Therefore, like most projects, PyDevice uses the deprecated functions, which still work. The drawback of these functions is that the Python runtime does not report an error when initialization fails. Instead, it terminates (crashes) the program.
  3) The 'stable' API does not support the buffer protocol for all Python versions, meaning that it is not possible to pass arrays to Python code except when explicitly relying on the `numpy` C API, which is very sensitive to a correct configuration of the Python paths and Python version, and version of the C-API (see [here](https://numpy.org/devdocs/user/troubleshooting-importerror.html)). 

To work around all the issues above, PyDevice imports the python runtime dynamically (known as delay loading). This way, the Python runtime is loaded only when the plugin is loaded, and the user can specify which version of Python to use exactly. Here are the steps that PyDevice takes to load the runtime:

1) First, the `PythonEnvironment` is used to locate the correct virtual environment or global installation. 
2) The `home` entry in the `pyvenv.cfg` configuration file in that virtual environment is then used to locate the Python runtime, allowing the correct version of Python to be used.
3) The `pythonXX.dll` runtime from that location is loaded dynamically using `LoadLibrary`. 

To facilitate delay-loading the dll, the `DELAYLOAD` support of Visual Studio is used. This way, there is almost no overhead in the c++ code. Instead of using `Python.h`, a list of function signatures for the stable API is included in `stable.h`, and functions for the buffer protocol are included in `buffer.h`. Have a look at `stable.cpp` to see how the Python runtime is loaded. For this automatic delay loading to work, the linker needs access to the file `python39.lib`. Note that this file is not actually used for linking anything, only to provide the linker with the information about which functions are present in the delay-loaded Python dll. Therefore, the result will not depend on the version of Python that was used to build the plugin.

### Debugging

To debug the plugin in Micro-Manager, select a `Debug` build configuration in Visual Studio. In addition, right-click
PyDevice in the solution explorer, and `Set as Startup Project`. Right-click again and edit the PyDevice project
Properties, under Debugging, fill in the following settings:

|                   |                                                         |
|-------------------|---------------------------------------------------------|
| Command           | $(ProjectDir)/debug_helper.bat                          |
| Command Arguments | $(TargetPath)                                           |
| Environment       | MM_EXECUTABLE_PATH="C:\Program Files\Micro-Manager-2.0" |

Here, adjust the `MM_EXECUTABLE_PATH` if you installed Micro-Manager somewhere else. Finally, in the menu select `Debug->Other Debug Targets->Child Process Debug Settings...` and tick `Enable child process debugging`.
You can now press F5 to build and install the plugin, and start Micro-Manager with the debugger attached. Note: you can safely ignore the segmentation faults occurring in the Java virtual machine.

For device scripts, we recommend debugging and testing the scripts outside Micro-Manager first. The scripts should simply as standalone Python scripts, since they do not contain any Micro-Manager specific code. This way, you can test the scripts in an environment that is easier to debug and test. After testing the devices, you can also load them in `pymmcore` to test the interaction with Micro-Manager while maintaining the ability to debug the Python code.

Finally, it is also possible to debug the c++ code while running pymmcore. To do so, attach the debugger to your Python IDE (e.g. PyCharm). This way, you can set breakpoints in the c++ code and inspect the state of the program while running pymmcore.

When debugging PyDevice, be aware that any changes to the c++ code, or to `bootstrap.py`, require the code to be re-compiled and the device adapter to be installed again (which is done automatically by the debug helper). At the moment, Visual Studio does not detect changes in `bootstrap.py` and does not recompile the plugin when this file is changed. Therefore, you need to manually rebuild the device adapter in Visual Studio to see the changes. 

