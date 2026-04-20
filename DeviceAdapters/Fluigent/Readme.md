# Micro-Manager Device Adapter for Fluigent Pressure controllers
*by Lars Kool, Institut Pierre-Gilles de Gennes (IPGG), Paris, France*

This repository contains the implementation of a $\mu$Manager pressure controller for Fluigent pressure controllers (LineUp series and MFCS series). The device adapter is tested on windows with multiple MFCS 0-1 bar (EZ-01000001), but it should work with all Fluigent pressure controllers.

## Getting started
1. The first step is to download the driver from the Fluigent Github repository, which can be found [here](https://github.com/Fluigent/fgt-SDK).
2. Next, copy the Fluigent driver (which can be found in the Github repository at "FLUIGENT_GITHUB_REPOSITORY/C++/fgt_SDK_Cpp/dlls/YOUR_OS/x64/fgt_SDK.dll) to the $\mu$Manager install folder (typically C:\Program Files\Micro-Manager-2.0).
3. The Fluigent devices are ready for action!
4. Start $\mu$Manager and open the Hardware Configuration Wizard, and Add the FluigentControllerHub, which can be found under Fluigent.
5. The device adapter will automatically detect all connected Fluigent devices, and will prompt you to select the desired pressure outlets.
6. Finish the Hardware Configuration (the Fluigent device adapter does not require any more input).
7. You can now control the pressure of each added channel using the Device Property Browser, Beanshell scripts, or by developing a small user interface.

## Features
- Change displayed pressure unit: PSI, kPa, Pa, bar, mbar (all pressures are converted to kPa internally)
- Control pressure (Imposed Pressure)
- Read out actual pressure (Measured Pressure)
- Activate internal calibration (channel-by-channel, or all at once)

## Limitations
- Fluigent supports indirect flow-rate control using their flow sensors. This features is however not (yet) supported by this driver.

## Future features
- Support for Fluigent valves
- Support for Fluigent flow sensors

Note that these are just ideas, no promises are made that these will be implemented in a timely manner (or at all). Other suggestions are more than welcome, either create a github issue or send an email to lars.kool@espci.fr