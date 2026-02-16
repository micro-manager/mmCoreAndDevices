# iSIMWaveforms Requirements

## Behavior

### General Behavior

- Interpret the Micro-Manager exposure time as the frame interval
  - Consistent with Rolling Shutter mode in PVCAM

### Pre-Initialization

- NIDAQ device and analog output channel discovery happens first
- Device initialization fails with helpful error message if required pre-init properties are unset

### Post-Initialization

#### Live or Snap Clicked (Normal Mode Imaging)

- If no MOD IN channels are enabled, only build the camera trigger waveform
- If any MOD IN channel is enabled, build the galvo, camera, blanking and MOD in waveforms
  - all enabled MOD IN pulses are high during every frame's exposure (no interleaving)
  
#### MDA (Interleaved Imaging)

- If one channel is selected in the MDA, pulse high the corresponding MOD IN channel every frame
- If multiple channels are selected interleave the MOD IN pulses so that the first channel is high
  on frame 1, the second on frame 2, etc. and repeat
    - Blanking is always high during exposure
	- Use the blanking and MOD IN voltage values that were set in the device property browser
	
#### Events that Trigger Waveform Rebuilds

The following will trigger a waveform rebuild:

- Camera exposure time changes
- Camera readout time changes
- Any AOTF channel voltage value changes
- Any galvo waveform property changes, i.e. Vpp exposure, offset voltage, parking fraction

Some of these will be triggered by other events, e.g. the readout time might change when the ROI
size changes, or the exposure time might change when the physical camera changes

## Properties

### Pre-Initialization

The user can set the following properties:

- AOTF blanking analog output (AO channel with min and max voltage
  - Required
- AOTF MOD IN AO channels 1 through four with their min and max voltages
  - At least one is required
- Camera trigger channel with min and max voltages
  - Required
- Clock source channel
- Counter channel
- NIDAQ device (Dev1, Dev2, etc.)
  - Required
- Galvo waveform AO channel with min and max voltages
  - Required
- The physical camera readout time property name
  - Enter `None` to manually set the readout time after device initialization
  - Otherwise, enter the device property name for the readout time of the physical camera
- The conversion factor from the physical camera readout time's units to milliseconds
  - For example, the string `1e-6` converts nanoseconds to milliseconds
  
 ### Post-Initialization
 
 - AOTF Blanking Voltage (float)
 - AOTF MOD IN 1-4 Enabled (boolean)
 - Binning (integer)
   - Required only because we implement MMCamera
 - Exposure peak-to-peak voltage (float)
 - Galvo offset voltage (float)
 - Parking fraction (float)
 - Physical camera (string; dropdown selection)
 - Readout time property name (string; read only)
 - Readout time (float; read only)
 - Sampling rate in Hz (float)