## PyDevice; The MicroManager device to load Python scripts as MicroManager devices.

## How does it work?
 - The PyDevice runs a Python interpreter in which the Python functions are executed. 
   Due to following specific naming conventions, the properties and methods of objects can be translated to MicroManager properties and methods

## How do I build it? 
 - We are hoping to integrate this device into the main MicroManager distribution, in that case, it might already be available to you.
   Else: Read the BUILD_INSTRUCTION.md file. 

## How do I use it?
 - If the device is available in your MicroManager install, you can test it by loading the test.py file. 
   When adding the PyHub in the config. manager, the boxes 'ModulePath' and 'ScriptPath' appear. ModulePath signifies the location of your Python  
   home. This can be a virtual environment, an Anaconda install or just a Python install, as long as its Python 3.
 
   If you leave the ModulePath on '(auto)' it will most likely find a Python install for you. If the script you load is in a virtual environment,
   It will find the virtual environment for you. This is important as it contains specific packages (like NumPy) that your scripts need to run.

## How do I build my own devices?
 - We have clarified that in more detail in the subfolder Python_devices,  starting with HOW_TO_MAKE_A_DEVICE.md. 