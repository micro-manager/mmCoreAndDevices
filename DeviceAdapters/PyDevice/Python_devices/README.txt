::In this folder are the examples of devices that can be loaded into MicroManager.

::To test one out without hardware, the test.py is reccomended.

::This folder contains:
- gain.py: A device responsible for controlling the gain of a PMT controller using a Nidaq DAC.
- galvo_scanner.py: A device responsible for laser scanning using a Nidaq DAC.
- Pyscanner.py: The functions neccesary for galvo_scanner.py, Cannot be loaded into micromanager
- test.py: Example camera that creates an image with random noise and a device that controls the properties of that noise.
