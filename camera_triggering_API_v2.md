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

```c++
////////////////////////////////////////////
// Micro-Manager Camera API proposal (v2)
////////////////////////////////////////////


//////////////////////////////
// trigger states
//////////////////////////////

// TODO: change from enums to ints

enum TriggerTypes { AcquisitionStart, AcquisitionEnd, AcquisitionActive, FrameBurstStart, FrameBurstEnd, FrameBurstActive, 
                  FrameStart, FrameEnd, FrameActive, ExposureStart, ExposureEnd and ExposureActive}

enum TriggerMode {On, Off}

// "external" -- TTL pulse
// "software" -- a call from "TriggerSoftware" function
// "internal" -- From the cameras internal timer
enum TriggerSource {Software, External, Internal}       
 
enum TriggerActivation{RisingEdge, FallingEdge, AnyEdge, LevelHigh, LevelLow}
 
// Off: No trigger overlap is permitted.
// ReadOut: Trigger is accepted immediately after the exposure period.
// PreviousFrame: Trigger is accepted (latched) at any time during the capture of the previous frame.
enum TriggerOverlap{Off, ReadOut, PreviousFrame }

// Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.
float TriggerDelay_us;

//////////////////////////////
// trigger functions
//////////////////////////////
      

//Check which of the possible trigger types are available
int hasTriggerType(TriggerType)

// These should return an error code if the type is not valid
// They are not meant to do any work. 
int setTriggerState(TriggerType, TriggerMode, TriggerSource);
int setTriggerState(TriggerType, TriggerMode, TriggerSource, TriggerDelay, TriggerActivation, TriggerOverlap);
 
int getTriggerState(&TriggerType, &TriggerMode, &TriggerSource);
int getTriggerState(&TriggerType, &TriggerMode, &TriggerSource, &TriggerDelay, &TriggerActivation, &TriggerOverlap);


// Send of software of the supplied type
TriggerSoftware(TriggerType)



//////////////////////////////
// acquisition parameters
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



//////////////////////////////
// acquisition functions
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
int AcquisitionAbort()






// Maybe: for querying acquistion status
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
void setRollingShutterLineOffset(double offset_us) throw (CMMError);

int GetRollingShutterActiveLines();
void setRollingShutterActiveLines(int numLines) throw (CMMError);
```


## Changes to Core Callback API
TODO trigger change callback

// This generalizes the prepareForAcq and acqFinished
//TODO what are the types of events
cameraEventCallback(int ev)


## New calls in MMCore
A set of API calls in MMCore will provide access to this high-level API. Following MM convention, these will be essentially a 1to1 access of camera API methods, so they are omitted for now.


## Backwards compatibility
TODO...


## Data access

The current MMCore provides two routes for accessing image data, one for Snaps and one for Sequences. The newer API will eventually be used with a unified, single image storage mechanism. Thus, the new APIs will always insert images into the circular buffer. For backwards compatibility, when `core.SnapImage` is called, the image will be copied into a seperate buffer so that it can be retrieved in the expected way for old API users.

