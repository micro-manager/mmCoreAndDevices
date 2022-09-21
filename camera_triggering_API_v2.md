# A proposed new API for camera triggering

Based on the [GenICam specification](https://www.emva.org/wp-content/uploads/GenICam_SFNC_2_2.pdf), with modifications as needed for consistency with the existing API

`internal` triggers mean no signal is sent for the corresponding event (the same triggers being "off" in GenICam)
`external` triggers mean a TTL pulse is sent to the camera
`software` triggers mean an API function is called

**Development discussions:**

[Chapter 1](https://github.com/micro-manager/mmCoreAndDevices/issues/84)

[Chapter 2](https://github.com/micro-manager/mmCoreAndDevices/issues/226)

**Previous API proposal versions:**

[v1](https://github.com/micro-manager/mmCoreAndDevices/blob/4ce6d521a3875c0ecdd691fba5751b6a43bb0cd1/camera_triggering_API.md)


## Changes to [MM::Camera](https://valelab4.ucsf.edu/~MM/doc/MMDevice/html/class_m_m_1_1_camera.html)

```c++
////////////////////////////////////////////
// Micro-Manager Camera API proposal (v2)
////////////////////////////////////////////


bool isNewAPIImplemented();


//////////////////////////////
// Triggers
//////////////////////////////


// trigger state constants
//////////////////////////////

//  TriggerSelector
const int TriggerSelectorAcquisitionStart = 0;
const int TriggerSelectorAcquisitionEnd = 1;
const int TriggerSelectorAcquisitionActive = 2;
const int TriggerSelectorFrameBurstStart = 3;
const int TriggerSelectorFrameBurstEnd = 4;
const int TriggerSelectorFrameBurstActive = 5;
const int TriggerSelectorFrameStart = 6;
const int TriggerSelectorFrameEnd = 7;
const int TriggerSelectorFrameActive = 8;
const int TriggerSelectorExposureStart = 9;
const int TriggerSelectorExposureEnd = 10;
const int TriggerSelectorExposureActive = 11;


// TriggerMode
const int TriggerModeOn = 0;
const int TriggerModeOff = 1;


// TriggerSource
//  "internal" -- From the cameras internal timer
//  "external" -- TTL pulse
//  "software" -- a call from "TriggerSoftware" function
const int TriggerSourceInternal = 0;
const int TriggerSourceExternal = 1;
const int TriggerSourceSoftware = 2;


// TriggerActivation
const int TriggerActivationAnyEdge = 0;
const int TriggerActivationRisingEdge = 1;
const int TriggerActivationFallingEdge = 2;
const int TriggerActivationLevelLow = 3;
const int TriggerActivationLevelHigh = 4;

 
// TriggerOverlap
//  Off: No trigger overlap is permitted.
//  ReadOut: Trigger is accepted immediately after the exposure period.
//  PreviousFrame: Trigger is accepted (latched) at any time during the capture of the previous frame.
const int TriggerOverlapOff = 0;
const int TriggerOverlapReadout = 1;
const int TriggerOverlapPreviosFrame = 2;


// trigger functions
//////////////////////////////
      

//Check which of the possible trigger types are available
int hasTrigger(int triggerSelector);

// These should return an error code if the type is not valid
// They are not meant to do any work. 
int setTriggerState(int triggerSelector, int triggerMode, int triggerSource);
int setTriggerState(int triggerSelector, int triggerMode, int triggerSource, int triggerDelay, int triggerActivation, int triggerOverlap);
 
int getTriggerState(int &triggerSelector, int &triggerMode, int &triggerSource);
int getTriggerState(int &triggerSelector, int &triggerMode, int &triggerSource, int &triggerDelay, int &triggerActivation, int &triggerOverlap);


// Send of software of the supplied type
int TriggerSoftware(int triggerSelector);



//////////////////////////////
// Acquisitions
//////////////////////////////

// Some terminology form GenICam
//
// AcquisitionMode
//    SingleFrame: One frame is captured.
//    MultiFrame: The number of frames specified by AcquisitionFrameCount is captured.
//    Continuous: Frames are captured continuously until stopped with the AcquisitionStop command.
//
// AcquisitionFrameCount
//    Number of frames to acquire in MultiFrame Acquisition mode. 
//
// AcquisitionBurstFrameCount 
//    Number of frames to acquire for each FrameBurstStart trigger.
//    This feature is used only if the FrameBurstStart trigger is enabled and
//    the FrameBurstEnd trigger is disabled. Note that the total number of frames
//    captured is also conditioned by AcquisitionFrameCount if AcquisitionMode is
//    MultiFrame and ignored if AcquisitionMode is Single.
//
// AcquisitionFrameRate
//    Controls the acquisition rate (in Hertz) at which the frames are captured.
//    TriggerMode must be Off for the Frame trigger.




// Acquisition functions
//////////////////////////////

// Arms the device before an AcquisitionStart command. This optional command validates all 
// the current features for consistency and prepares the device for a fast start of the Acquisition.
// If not used explicitly, this command will be automatically executed at the first 
// AcquisitionStart but will not be repeated for the subsequent ones unless a feature is changed in the device.

// TODO: the above logic needs to be implemented in core?

// Don't acqMode because it can be inferred from frameCount
// if frameCount is:    1 --> acqMode is single
//                    > 1 --> acqMode is MultiFrame
//                     -1 --> acqMode is continuous

int AcquisitionArm(int frameCount, double acquisitionFrameRate, int burstFrameCount);
int AcquisitionArm(int frameCount, int burstFrameCount);
int AcquisitionArm(int frameCount, double acquisitionFrameRate);
int AcquisitionArm(int frameCount);



// Starts the Acquisition of the device. The number of frames captured is specified by AcquisitionMode.
// Note that unless the AcquisitionArm was executed since the last feature change, 
// the AcquisitionStart command must validate all the current features for consistency before starting the Acquisition. 
int AcquisitionStart();

// Stops the Acquisition of the device at the end of the current Frame. It is mainly 
// used when AcquisitionMode is Continuous but can be used in any acquisition mode.
// If the camera is waiting for a trigger, the pending Frame will be cancelled. 
// If no Acquisition is in progress, the command is ignored.
int AcquisitionStop();


//Aborts the Acquisition immediately. This will end the capture without completing
// the current Frame or waiting on a trigger. If no Acquisition is in progress, the command is ignored.
int AcquisitionAbort();



// Maybe: for querying acquisition status
// AcquisitionTriggerWait: Device is currently waiting for a trigger for the capture of one or many frames.
// AcquisitionActive: Device is currently doing an acquisition of one or many frames.
// AcquisitionTransfer: Device is currently transferring an acquisition of one or many frames.
// FrameTriggerWait: Device is currently waiting for a frame start trigger.
// FrameActive: Device is currently doing the capture of a frame.
// ExposureActive: Device is doing the exposure of a frame.

enum AcquisitionStatusType = {AcquisitionTriggerWait, AcquisitionActive, AcquisitionTransfer, FrameTriggerWait, FrameActive, ExposureActive}
bool readAcquisitionStatus(AcquisitionStatusType a);




// Rolling shutter/Lightsheet mode
double GetRollingShutterLineOffset();
void SetRollingShutterLineOffset(double offset_us) throw (CMMError);

int GetRollingShutterActiveLines();
void SetRollingShutterActiveLines(int numLines) throw (CMMError);
```


## Changes to [Core (Callback) API](https://valelab4.ucsf.edu/~MM/doc/MMDevice/html/class_m_m_1_1_core.html)
```c++
// Called by camera when trigger changes
OnCameraTriggerChanged (const Device *caller, int triggerSelector, int triggerMode, int triggerSource);
OnCameraTriggerChanged (const Device *caller, int triggerSelector, int triggerMode, int triggerSource, int triggerDelay, int triggerActivation, int triggerOverlap);


// Callbacks for camera events. This generalizes the prepareForAcq and acqFinished

// AcquisitionTrigger: Device just received a trigger for the Acquisition of one or many Frames.
// AcquisitionStart: Device just started the Acquisition of one or many Frames.
// AcquisitionEnd: Device just completed the Acquisition of one or many Frames.
// AcquisitionTransferStart: Device just started the transfer of one or many Frames.
// AcquisitionTransferEnd: Device just completed the transfer of one or many Frames.
// AcquisitionError: Device just detected an error during the active Acquisition.
// FrameTrigger: Device just received a trigger to start the capture of one Frame.
// FrameStart: Device just started the capture of one Frame.
// FrameEnd: Device just completed the capture of one Frame.
// FrameBurstStart: Device just started the capture of a burst of Frames.
// FrameBurstEnd: Device just completed the capture of a burst of Frames.
// FrameTransferStart: Device just started the transfer of one Frame.
// FrameTransferEnd: Device just completed the transfer of one Frame.
// ExposureStart: Device just started the exposure of one Frame (or Line).
// ExposureEnd: Device just completed the exposure of one Frame (or Line).
const int CameraEventAcquisitionTrigger = 0;
const int CameraEventAcquisitionStart = 1;
const int CameraEventAcquisitionEnd = 2;
const int CameraEventAcquisitionTransferStart = 3;
const int CameraEventAcquisitionTransferEnd = 4;
const int CameraEventAcquisitionError = 5;
const int CameraEventFrameTrigger = 6;
const int CameraEventFrameStart = 7;
const int CameraEventFrameEnd = 8;
const int CameraEventFrameBurstStart = 9;
const int CameraEventFrameBurstEnd = 10;
const int CameraEventFrameTransferStart = 11;
const int CameraEventFrameTransferEnd = 12;
const int CameraEventExposureStart = 13;
const int CameraEventExposureEnd = 14;

// Camera calls this function on the core to notify of events
// TODO: this may be coming from a camera internal thread,
// so it may make sense to put restrictions on what the core
// can do with these callbacks (i.e. nothing processor intensive)
cameraEventCallback(const Device *caller, int EventType)
```

## New calls in [MMCore](https://valelab4.ucsf.edu/~MM/doc/MMCore/html/class_c_m_m_core.html)
A set of API calls in MMCore will provide access to this high-level API. Following MM convention, these will be essentially a 1to1 access of camera API methods.

TODO...

## Backwards compatibility
The old (camera) API for now will be optional on new devices, to be removed later in the future (maybe)

The old (core) API will be implemented in terms of the new camera API when it is present, or fall back on the old camera API when its not

## Data access

The current MMCore provides two routes for accessing image data, one for Snaps and one for Sequences. The newer API will eventually be used with a unified, single image storage mechanism. Thus, the new APIs will always insert images into the circular buffer. For backwards compatibility, when `core.SnapImage` is called, the image will be copied into a seperate buffer so that it can be retrieved in the expected way for old API users.

