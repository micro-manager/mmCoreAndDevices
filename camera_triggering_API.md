# A proposed new API for camera triggering

Based largerly off of the [GenICam specification](https://www.emva.org/wp-content/uploads/GenICam_SFNC_2_2.pdf), with modifications as needed for consistency with the existing API

`internal` triggers mean no signal is sent for the corresponding event (the same triggers being "off" in GenICam)
`external` triggers mean a TTL pulse is sent to the camera
`software` triggers mean an API function is called

previous development of this API is discussed [here](https://github.com/micro-manager/mmCoreAndDevices/issues/84), but this file contains the most up to date version.

```c++
////////////////////////////////////////////
// Call these methods first to describe the desired trigger mode 
////////////////////////////////////////////

// Control delays
int SetPreFrameDelay(double time_ms);
int SetPostFrameDelay(double time_ms);

// "external" (TTL pulse), "software" (API call), "internal" (Camera is always ready for a burst)
int SetBurstStartTriggerType(const char * type);

// "external" (TTL pulse), "software" ("API call"), or "internal" (same as above)
int SetBurstEndTriggerType(const char * type);

// "external" (TTL pulse),  "software" (API call), "internal" (camera handles based on exposure/delays)
int SetFrameStartTriggerType(const char * type);

// "external" (TTL pulse),  "software" (could be API call, isn't currently, unclear why'd you ever use it), 
// "internal"  (based on camera.SetExposure(double ms))
int SetExposureEndTriggerType(const char * type);

//// Special conditions not covered by above settings
// "distinct" (start + end TTLs), "exposure" (bulb/strobe TTL), or "combined" (synchronous TTLs)
// Only applicable with "external" FrameStartTrigger/ExposureEndTrigger
int SetFrameExposureMode(const char * mode);

////////////////////////////////////////////
// Running a burst
////////////////////////////////////////////

// Arm the camera so that its ready to receive a BurstStartTrigger
// If the requested combination of trigger settings does not make sense is nnot supported, throw exception
// Needs to know the number or images if using  "internal" BurstEndTrigger (i.e. camera stops its own burst)
// If numImages == -1, run until an explicit BurstEndTrigger from API call ("software") or TTL pulse ("external")
int PrepareForBurst(int numImages);


////////////////////////////////////////////
// Software Triggering
////////////////////////////////////////////
// These apply when using "software" triggers of the corresponding type
int SendBurstStartTrigger();
// This method blocks for the duration of exposure, so that shutters can be closed immediately upon returning
int SendFrameStartTrigger(); // alias for SnapImage();
int SendBurstEndTrigger();



// Rolling shutter/Lightsheet mode
double GetRollingShutterLineOffset();
void setRollingShutterLineOffset(double offset_us) throw (CMMError);

int GetRollingShutterActiveLines();
void setRollingShutterActiveLines(int numLines) throw (CMMError);
```


## New calls in MMCore
A set of API calls in MMCore will provide access to this high-level API. Following MM convention, these will be essentially a 1to1 access of camera API methods, so they are omitted for now.


## Backwards compatibility
The core can be updated to use the new camera API where available, yet still maintain backwards compatibility with the current core API by implementing the old Core API with new Camera API calls. This allows for new cameras to implement only the new MMDevice API, but still be backwards compatible with the existing MMCore functions. For cameras that don't implement the newer API, cameras will fall back to the current way.

For example, when `core.snapImage()` is called and a device only has the new API implemented, the following will happen:

```c++
camera.SetBurstStartTriggerType("internal"); // not needed
camera.SetFrameStartTriggerType("software"); // API call below
camera.SetExposureEndTriggerType("internal"); // Determined previously by SetExposure(double exposure)
camera.SetBurstEndTriggerType("internal"); // Determined by numImages 

camera.PrepareForBurst(1); // 1 image

// No burst start trigger needed
camera.SendFrameStartTrigger();
// ExposureEnd and BurstEnd "internally" triggered based on exposure time and numImages, respectively
```


```core.startContinuousSequenceAcquisition(double ms)```

will under the hood call:

```c++
// Assuming SetPreFrameDelay(double ms) and SetPostFrameDelay(double ms) 
// delays have already been set, which along with exposure control the frame rate
camera.SetBurstStartTriggerType("software"); 
camera.SetFrameStartTriggerType("internal");
camera.SetExposureEndTriggerType("internal"); // Determined previous call to SetExposure(double exposure);
camera.SetBurstEndTriggerType("software"); // Go until API call to stop

camera.PrepareForBurst(-1); // Go indefinitely
camera.SendBurstStartTrigger();
```

```core.stopSequenceAcquisition()```

will call:

```c++
camera.SendBurstEndTrigger();
```


The only one that doesn't map neatly is:

```c++
core.prepareSequenceAcquisition()
```

which unfortunately doesn't give the number of images needed. So the two lines:



```c++
core.prepareSequenceAcquisition();
core.startSequenceAcquisition (long numImages,
      double intervalMs, bool stopOnOverflow); //these two are generally ignored already
```

is called as:

```c++
camera.SetBurstStartTriggerType("software"); 
camera.SetFrameStartTriggerType("internal"); //determined by number of images
camera.SetExposureEndTriggerType("internal"); // Determined previous call to SetExposure(double exposure);
camera.SetBurstEndTriggerType("internal"); // Will stop after number of images have been acquired

camera.PrepareForBurst(numImages);
camera.SendBurstStartTrigger();
```

This will mean, essentially that when the new camera API is implemented, `core.prepareSequenceAcquisition()` will be ignored. As result, higher level code should preferentially utilize the new API to achieve maximium performance.

## Data access

The current MMCore provides two routes for accessing image data, one for Snaps and one for Sequences. The newer API will eventually be used with a unified, single image storage mechanism. Thus, the new APIs will always insert images into the circular buffer. For backwards compatibility, when `core.SnapImage` is called, the image will be copied into a seperate buffer so that it can be retrieved in the expected way for old API users.

