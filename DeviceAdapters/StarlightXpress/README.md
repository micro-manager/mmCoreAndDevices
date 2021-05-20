# StarlightXpress Filter Wheel Device Adapters
|                                |                                     |
| ------------------------------ | ----------------------------------- |
| **Summary**                    | Starlight Xpress Device Adapters    |
| **Authors**                    | Elliot Steele                       |
| **License**                    | MIT                                 |
| **Platforms**                  | Windows, Mac, Linux                 |
| **Devices**                    | Filter Wheels*                      |
| **Manufacturer Website**       | https://www.sxccd.com/filterwheels/ |

\* Currently only tested with the MIDI filter wheels

## Introduction
These device adapters provide support for filter wheels from Starlight Xpress Ltd.

## Driver installation 
Communication is via the HID USB interface and as such no specific driver installation is required on most operating systems. On linux systems utilising udev for handling device events (including Ubuntu), an appropriate udev rules file may be required to allow Micro-Manager access to the device.

## Setup
**Make sure the Filter Wheel has finished its startup sequence before attempting to connect**

The device adapter relies on the HIDManager device adapter to make the USB interface appear as a serial port to the main Starlight Xpress adapters. Because of this the adapters may also work via the filter wheel's dedicated serial interface but this is untested. The port pre-init property is used to specify which device to connect to. Thanks to the HIDManager, this should appear as "StarlightXpressFilterWheel" when connected. Other pre-init properties are described in the next section.

## Properties
### Filter Calibration Mode and Number of Filters (Pre-Init)
The Starlight Xpress filter wheel has a calibration feature that allows it to determine the number of filter positions automatically. When the "Filter Calibration Mode" property is set to "Auto", the calibration routine will be run whenever the Micro-Manager configuration is loaded. 

Unfortunately, this routine is slow and seems to fail to determine the correct number of filter wheels intermittently. The calibration can therefore be bypassed by setting the "Calibration Mode" property to "Manual". When in manual mode, the number of filters is determined by the "Number of Filters" property. When the number of filters is set to 0 in manual mode, the device adapter will request the number of filters from the wheel *without* running the calibration routine (**Note:** this may cause the incorrect number of filters to be returned). Special care should be taken not to set the number of filters to greater than the actual number of positions as requesting a filter position greater than the maximum number of positions (e.g., position 6 on a 5 position wheel) may cause the wheel to spin indefinitely.

### Poll Delay (ms)
The poll delay is the time the device adapter waits between requesting a filter change and querying the new filter position. Idealy, this should be set to approximately the time it takes for the wheel to switch between adjacent filters. The poll delay is multiplied by the distance moved, i.e., moving from filter position 1 to 3, a distance of 2 filters, would use a delay of 2x(poll delay). Setting this value too low may cause the wheel to freeze.