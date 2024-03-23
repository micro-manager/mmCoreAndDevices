///////////////////////////////////////////////////////////////////////////////
// FILE:          PVCAMAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PVCAM camera module
//                
// AUTHOR:        Nico Stuurman, Nenad Amodaj nenad@amodaj.com, 09/13/2005
// COPYRIGHT:     University of California, San Francisco, 2006
//                100X Imaging Inc, 2008
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//                Micromax compatible adapter is moved to PVCAMPI project, N.A. 10/2007
//
// CVS:           $Id: PVCAM.h 8240 2011-12-04 01:05:17Z nico $

#ifndef _PVCAMADAPTER_H_
#define _PVCAMADAPTER_H_


//=============================================================================
//==================================================================== INCLUDES


// MMDevice
#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "DeviceUtils.h"
#include "ImgBuffer.h"

// Local
#include "AcqConfig.h"
#include "Event.h"
#include "NotificationEntry.h"
#include "PVCAMIncludes.h"
#include "PvCircularBuffer.h"
#include "PvDebayer.h"
#include "PpParam.h"

// System
#include <map>
#include <memory> // smart pointers
#include <string>
#include <utility> // std::pair


//=============================================================================
//===================================================================== DEFINES


// FRAME_INFO support (on Windows since PVCAM 2.9.5, on Linux since 3.0.4)
#define PVCAM_FRAME_INFO_SUPPORTED
// Callbacks ex3 support (on Windows since PVCAM 2.8.1, on Linux since 3.0.4)
#define PVCAM_CALLBACKS_SUPPORTED
// The new parameter support (on Windows since PVCAM 3.0.0, on Linux since 3.0.4)
#define PVCAM_PARAM_EXPOSE_OUT_DEFINED
// The SMART streaming support (on Windows since PVCAM 2.8.0, on Linux since 3.0.4)
#define PVCAM_SMART_STREAMING_SUPPORTED
// Metadata, Multi-ROI, Centroids and other features that were added to PVCAM 3.0.12
#define PVCAM_METADATA_SUPPORTED
// Software trigger support (since PVCAM 3.9.0)
#define PVCAM_SW_TRIGGER_SUPPORTED

// PVCAM 3.1+ has some additional PL_COLOR_MODES defined which we use across the code
// even if we don't compile against that PVCAM. To make it easier we define them ourselves.
#ifndef PVCAM_METADATA_SUPPORTED
#define COLOR_GRBG 3
#define COLOR_GBRG 4
#define COLOR_BGGR 5
#endif

// PVCAM 3.9+ has some additional PL_EXPOSURE_MODES defined which we use across the code
// even if we don't compile against that PVCAM. To make it easier we define them ourselves.
#ifndef PVCAM_SW_TRIGGER_SUPPORTED
#define EXT_TRIG_SOFTWARE_FIRST ((7 + 4) << 8)
#define EXT_TRIG_SOFTWARE_EDGE  ((7 + 5) << 8)
#endif

// Custom error codes originating from this adapter
#define ERR_INVALID_BUFFER              10002
#define ERR_INVALID_PARAMETER_VALUE     10003
#define ERR_BUSY_ACQUIRING              10004
#define ERR_STREAM_MODE_NOT_SUPPORTED   10005
#define ERR_CAMERA_NOT_FOUND            10006
#define ERR_ROI_SIZE_NOT_SUPPORTED      10007
#define ERR_BUFFER_TOO_LARGE            10008
#define ERR_ROI_DEFINITION_INVALID      10009
#define ERR_BUFFER_PROCESSING_FAILED    10010
#define ERR_BINNING_INVALID             10011 // Binning value is not valid for current configuration
#define ERR_OPERATION_TIMED_OUT         10012 // Generic timeout error
#define ERR_FRAME_READOUT_FAILED        10013 // Polling: status = READOUT_FAILED
#define ERR_TOO_MANY_ROIS               10014 // Device does not support that many ROIs (uM 2.0)
#define ERR_FILE_OPERATION_FAILED       10015
#define ERR_SW_TRIGGER_NOT_SUPPORTED    10016
#define ERR_PIXEL_TYPE_NOT_SUPPORTED    10017

// PVCAM-specific error codes base. When a PVCAM error occurs we use the PVCAM
// ID and PVCAM message to create a new uM error code, we call the SetErrorCode()
// and assign a text to a new error code value. In order to not interfere with existing
// error codes we need to add some offset.
// Example:
//     if (pl_something() != PV_OK)
//        SetErrorCode(pl_error_code() + ERR_PVCAM_OFFSET, pl_error_message());
#define ERR_PVCAM_OFFSET                20000

//=============================================================================
//=========================================================== TYPE DECLARATIONS

/**
* Structure used for Universal Parameters definition
*/
struct ParamNameIdPair
{
    const char* name;
    const char* debugName;
    uns32 id;
};

/**
* Speed table row
*/
struct SpdTabEntry
{
    uns16 pixTime;         // Readout rate in ns
    rs_bool gainAvail;     // Gain available
    int16   gainMin;       // Min gain index for this speed
    int16   gainMax;       // Max gain index for this speed
    int16   gainDef;       // Default gain for this speed
    std::map<std::string, int16> gainNameMap; // Gain names (i.e., "name:index" map)
    std::map<int16, std::string> gainNameMapReverse; // Reverse lookup map
    int16 spdIndex;           // Speed index
    int32 portValue;          // Port index
    int16 portDefaultSpdIdx;  // Default speed index for given port (applied when port changes)
    std::string spdString;    // A string that describes this choice in GUI
    std::string spdName;      // A string received from camera, empty if not supported
    int32       colorMask;    // Sensor color mask (PARAM_COLOR_MODE)
    std::string colorMaskStr; // Sensor color mask description (retrieved from PVCAM)
};

/**
* Camera Model is identified mostly by Chip Name. Most of the cameras and every
* unknown camera is treated as "Generic". PVCAM and this uM adapter is mostly
* camera-agnostic, however a couple of camera models may need special treatment.
*/
enum PvCameraModel
{
    PvCameraModel_Generic = 0,
    PvCameraModel_OptiMos_M1,
    PvCameraModel_Retiga6000C
};

//=============================================================================
//======================================================== FORWARD DECLARATIONS

class PollingThread;
class NotificationThread;
class AcqThread;
class StreamWriter;
template<class T> class PvParam;
class PvUniversalParam;
class PvEnumParam;

//=============================================================================
//========================================================== CLASS DECLARATIONS

/**
* Implementation of the MMDevice and MMCamera interfaces for all PVCAM cameras
*/
class Universal : public CCameraBase<Universal>
{
public: // Constructors, destructor
    Universal(short cameraId, const char* deviceName);
    ~Universal();

public: // MMDevice API
    int  Initialize();
    int  Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    bool GetErrorText(int errorCode, char* text) const;

public: // MMCamera API
    /**
    * Acquires a single frame and stores it in the internal buffer.
    * This command blocks the calling thread until the image is fully captured.
    */
    int SnapImage();
    const unsigned char* GetImageBuffer();
    const unsigned* GetImageBufferAsRGB32();
    unsigned GetImageWidth() const;
    unsigned GetImageHeight() const;
    unsigned GetImageBytesPerPixel() const; 
    long GetImageBufferSize() const;
    unsigned GetBitDepth() const;
    int GetBinning() const;
    int SetBinning(int binSize);
    double GetExposure() const;
    void SetExposure(double dExp);
    int IsExposureSequenceable(bool& isSequenceable) const;
    unsigned GetNumberOfComponents() const;

    int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
    int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
    int ClearROI();
    bool SupportsMultiROI();
    bool IsMultiROISet();
    int GetMultiROICount(unsigned& count);
    int SetMultiROI(const unsigned* xs, const unsigned* ys, const unsigned* widths, const unsigned* heights, unsigned numROIs);
    int GetMultiROI(unsigned* xs, unsigned* ys, unsigned* widths, unsigned* heights, unsigned* length);

    bool IsCapturing();

    /**
    * Micro-manager calls the "live" acquisition a "sequence". PVCAM calls this "continuous - circular buffer" mode.
    */
    int PrepareSequenceAcquisition();
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
    int StopSequenceAcquisition();

public: // Action handlers
    /**
    * Universal properties are automatically read from the camera and does not need a custom
    * value handler. This is useful for simple camera parameters that does not need special treatment.
    * So far only Enum and Integer values are supported. Other types should be implemented manually.
    */
    int OnUniversalProperty(MM::PropertyBase* pProp, MM::ActionType eAct, long index);

    /**
    * Gets or sets the current binning. Accepts and returns a string value
    * of "HxV" where H=horizontal and V=vertical binning
    */
    int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current horizontal binning.
    */
    int OnBinningX(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current vertical binning.
    */
    int OnBinningY(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the current exposure time, in milli seconds, floating point value.
    */
    int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets the current pixel type as string: "XXbit"
    */
    int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the current gain index.
    */
    int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the readout port. Change in readout port resets the speed which
    * in turn changes Gain range, Pixel time, Actual Gain, Bit depth and Read Noise.
    * see portChanged() and speedChanged().
    */
    int OnReadoutPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the readout speed. The available choices are obtained from the
    * speed table which is build in Initialize(). If a change in speed occurs we need
    * to update Gain range, Pixel time, Actual Gain, Bit depth and Read Noise.
    * See speedChanged().
    */
    int OnReadoutRate(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the camera-reported speed name for currently selected speed.
    * The available choices are obtained from the speed table which is build in Initialize().
    */
    int OnReadoutRateName(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the EM/Multiplier gain if supported.
    */
    int OnMultiplierGain(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual bit depth as it goes from the camera to PVCAM.
    */
    int OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual image format as it goes from the camera to PVCAM.
    */
    int OnImageFormat(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual image compression as it goes from the camera to PVCAM.
    */
    int OnImageCompression(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual bit depth as it goes from PVCAM to the app.
    */
    int OnBitDepthHost(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual image format as it goes from PVCAM to the app.
    */
    int OnImageFormatHost(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the actual image compression as it goes from PVCAM to the app.
    */
    int OnImageCompressionHost(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets the current camera sensor temperature in degrees Celsius.
    */
    int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the desired camera sensor temperature, in degrees Celsius.
    */
    int OnTemperatureSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the current PMode - i.e. Frame Transfer mode.
    */
    int OnPMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current ADC offset
    */
    int OnAdcOffset(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Get or sets the current Scan Mode.
    */
    int OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Get or sets the current Scan Mode Direction.
    */    
    int OnScanDirection(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Get or sets the current Scan Direction Reset mode (on/off).
    */
    int OnScanDirectionReset(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the scan line delay.
    */ 
    int OnScanLineDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets the current scan line time.
    */
    int OnScanLineTime(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current scan width
    */
    int OnScanWidth(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the current Trigger Mode - i.e. Internal, Bulb, Edge, etc.
    */
    int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * The TriggerTimeOut is used in WaitForExposureDone() to specify how long should we wait
    * for a frame to arrive. Increasing this value may help to avoid timeouts on long exposures
    * or when there are long pauses between triggers.
    */
    int OnTriggerTimeOut(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current Expose Out mode - i.e. First Row, Any Row, All Rows, etc.
    */
    int OnExposeOutMode(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the current number of sensor clear cycles.
    */
    int OnClearCycles(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the current sensor clear mode.
    */
    int OnClearMode(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Enables or disables the use of circular buffer. When disabled the live acquisition
    * runs as a repeated sequence (something like fast time-lapse). The PVCAM continuous
    * mode is not used.
    */
    int OnCircBufferEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Enables automatic size adjustment of the circular buffer based on image size
    * and other acquisition factors.
    */
    int OnCircBufferSizeAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * The size of the frame buffer. Increasing this value may help in a situation when
    * camera is delivering frames faster than MM can retrieve them.
    */
    int OnCircBufferFrameCount(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Enables or disables a "frame recovery" algorithm that was used with older PVCAMs
    * to reduce the number of lost frames with very fast sequence acquisitions. This may
    * not be relevant for PVCAM 3.1.9.1+ where the reliability of frame delivery was
    * greatly improved.
    */
    int OnCircBufferFrameRecovery(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Enables or disables the embedded frame metadata feature. Introduced with Prime
    * camera. When enabled the camera does not send RAW pixels anymore but the buffer
    * contains headers and metadata which requires decoding before the image can be
    * sent to MMCore.
    */
    int OnMetadataEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * In normal operation, the timestamp is reset upon camera power up.
    * Use this function and related parameter to reset the timestamp when needed.
    * The parameter is a write-only, write 'true' value to reset the timestamp.
    */
    int OnMetadataResetTimestamp(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Enables or disables the Centroids feature. Introduced with Prime camera under
    * "SmartLocate" name. Requires Metadata to be enabled. When enabled the camera
    * analyzes the frame and picks regions that are interesting. Only those regions
    * are then transferred back to host. User can only configure the region size
    * and number of regions the camera should produce.
    */
    int OnCentroidsEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured Centroids radius in pixels.
    * E.g. if 5 is set the resulting centroid will be a square of 11 x 11
    * pixels (5 x 2 + 1)
    */
    int OnCentroidsRadius(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured Centroids count.
    * E.g. if 100 is set the camera will produce 100 small regions.
    */
    int OnCentroidsCount(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured Centroids mode.
    * The default Locate mode sends each centroid as separate ROI.
    * The other modes send all pixel data in first extra ROI and mark all objects
    * in it with header-only ROIs.
    */
    int OnCentroidsMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured Centroids frame count for background removal.
    * E.g. if 10 is set the camera will use first 10 frames from the
    * acquisition for internal optimization of further processing.
    */
    int OnCentroidsBgCount(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured Centroids threshold.
    * It is a fixed-point real number in Q8.4 format.
    * E.g. the value 1234 (0x4D2) from camera means 77.2 (0x4D hex = 77 dec).
    * MM shows the raw camera value to the user.
    */
    int OnCentroidsThreshold(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Gets or sets the currently configured camera fan speed.
    */
    int OnFanSpeedSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the camera trigger signal multiplexing.
    */
    int OnTrigTabLastMux(MM::PropertyBase* pProp, MM::ActionType eAct, long trigSignal);

    /**
    * Enables or disables the color mode processing.
    */
    int OnColorMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Camera sensor mask - read only, reported by camera. Note that this
    * property may change with port/speed so we need a full property for it.
    * Example: RGGB, BRGB, etc.
    */
    int OnSensorCfaMask(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the Red scale factor for debayering algorithm.
    */
    int OnRedScale(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the Green scale factor for debayering algorithm.
    */
    int OnGreenScale(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the Blue scale factor for debayering algorithm.
    */
    int OnBlueScale(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently applied sensor mask for the debayering algorithm.
    * Please note that the sensor has some physical mask (reported in OnSensorCfaMask)
    * but when ROI is used the mask has to be adjusted based on ROI coordinates.
    * This handler sets the actually applied mask for the algorithm.
    */
    int OnAlgorithmCfaMask(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Enables or disables the automated CFA mask selection for the debayering
    * algorithm. The correct mask is selected based on ROI coordinates and other
    * factors.
    */
    int OnAlgorithmCfaMaskAuto(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently selected debayering algorithm (Nearest, Bilinear, etc.)
    */
    int OnInterpolationAlgorithm(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Enables or disables the streaming to disk feature.
    */
    int OnDiskStreamingEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the path where raw files get stored.
    */
    int OnDiskStreamingPath(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the forward ratio for core.
    */
    int OnDiskStreamingCoreSkipRatio(MM::PropertyBase* pProp, MM::ActionType eAct);

#ifdef PVCAM_CALLBACKS_SUPPORTED
    /**
    * Switches between Callbacks or Polling acquisition type.
    */
    int OnAcquisitionMethod(MM::PropertyBase* pProp, MM::ActionType eAct);
#endif

    /**
    * Post processing parameter handler. Post processing features and parameters are
    * read out from the camera dynamically. Based on the camera provided information
    * a list of MM properties is automatically generated.
    */
    int OnPostProcProperties(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
    /**
    * Resets all post processing parameters to their default values. This property
    * acts as a "button". User sets it to "ON" but it is automatically reverted back
    * to off.
    */
    int OnResetPostProcProperties(MM::PropertyBase* pProp, MM::ActionType eAct);

#ifdef PVCAM_SMART_STREAMING_SUPPORTED
    /**
    * Enables or disables the S.M.A.R.T streaming feature.
    */
    int OnSmartStreamingEnable(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Updates SMART streaming values based on user's input. User always enters the
    * values in milliseconds. Internally value is converted to microseconds if needed.
    */
    int OnSmartStreamingValues(MM::PropertyBase* pProp, MM::ActionType eAct);
#endif

    /**
    * Read-only: Shows the camera actual exposure time value in ns.
    */
    int OnTimingExposureTimeNs(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the camera actual readout time value in ns.
    */
    int OnTimingReadoutTimeNs(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the camera actual clearing time value in ns.
    */
    int OnTimingClearingTimeNs(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the camera actual pre-trigger delay value in ns.
    */
    int OnTimingPreTriggerDelayNs(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Read-only: Shows the camera actual post-trigger delay value in ns.
    */
    int OnTimingPostTriggerDelayNs(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * Enables or disables the Frame Summing feature on host side.
    * It is a feature provided by PVCAM and available for all cameras.
    */
    int OnHostFrameSummingEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured frame count to sum.
    */
    int OnHostFrameSummingCount(MM::PropertyBase* pProp, MM::ActionType eAct);
    /**
    * Gets or sets the currently configured output image format.
    * When summing multiple frames, the result pixel value can go quickly out of
    * 16-bit range. This allows to switch the output image format e.g. to 32-bit.
    */
    int OnHostFrameSummingFormat(MM::PropertyBase* pProp, MM::ActionType eAct);

public: // Other published methods
    /**
    * Returns the PVCAM camera handle.
    * Published to allow other classes access the camera.
    */
    short Handle();

    // All the logging methods below prepend the message with a PVCAM Adapter
    // specific prefix. We use that to unify the logs and to clearly see which logs
    // comes from the adapter. Usage of these functions is preferred over the generic
    // LogMessage() and similar.

    /**
    * Logs a PVCAM-specific error message, use this function ONLY and RIGHT AFTER a PVCAM call fails.
    * The function will automatically create a new uM-specific error code which can be returned to Core
    * and displayed in the UI. (see SetErrorMessage())
    * Usage:
    *     if (pl_someting() != PV_OK)
    *         LogPvcamError(...);
    * @param lineNr use __LINE__ macro
    * @param message Custom Log message. This should be a descriptive message similar to:
    *                "This call failed because of this", the PVCAM error code and message will be then
    *                appended to the message to create a final error message:
    *                "This call failed because of this. PVCAM Err:36, PVCAM Msg:'Cannot communicate with the camera'"
    * @param pvErrCode PVCAM error code, obtained with pl_error_code()
    * @param debug if true the message will be sent only in the log-debug mode
    */
    int   LogPvcamError(int lineNr, const std::string& message, int16 pvErrCode = pl_error_code(), bool debug = false) throw();

    /**
    * Logs an Adapter-specific or general MM error code (DEVICE_ERR, DEVICE_INTERNAL_INCONSISTENCY, etc).
    * Use this function to log and display an error message that does not directly originate from PVCAM.
    * @param mmErrCode MMCore specific error code or a custom error code created with SetErrorMessage()
    * @param lineNr use __LINE__ macro
    * @param message Descriptive error message
    * @param debug if true the message will be sent only in the log-debug mode
    */
    int   LogAdapterError(int mmErrCode, int lineNr, const std::string& message, bool debug = false) const throw();

    /**
    * Logs a simple message to the debug log. Includes file and line information.
    * @param lineNr use __LINE__ macro
    * @param debug if true the message will be sent only in the log-debug mode
    */
    void  LogAdapterMessage(int lineNr, const std::string& message, bool debug = true) const throw();

    /**
    * Logs a simple message to the debug log.
    * @param message Descriptive error message
    * @param debug if true the message will be sent only in the log-debug mode
    */
    void  LogAdapterMessage(const std::string& message, bool debug = true) const throw();

protected:
    /**
    * This method is called from the static PVCAM callback or polling thread.
    * The method should finish as fast as possible to avoid blocking the PVCAM.
    * If the execution of this method takes longer than frame readout + exposure,
    * the FrameAcquired for the next frame may not be called.
    */
    int FrameAcquired();
    /*
    * Pushes a final image with its metadata to the MMCore
    */
    int PushImageToMmCore(const unsigned char* pixBuffer, Metadata* pMd);
    /**
    * Called from the Notification Thread. Prepares the frame for insertion to the MMCore.
    */
    int ProcessNotification(const NotificationEntry& entry);

    int  PollingThreadRun(void);
    void PollingThreadExiting() throw();

private:
    // Make object non-copyable
    Universal(const Universal&) = delete;
    Universal& operator=(const Universal&) = delete;

    /**
    * Read and create basic static camera properties that will be displayed in
    * Device/Property Browser. These properties are read only and does not change
    * during camera session.
    */
    int initializeStaticCameraParams();
    /**
    * Initializes the "Universal" parameters. See also OnUniversalProperty().
    */
    int initializeUniversalParams();
    /**
    * Initializes the post processing features.
    */
    int initializePostProcessing();
    /**
    * Builds the speed table based on camera settings. We use the speed table to get actual
    * bit depth, readout speed and gain range based on speed index.
    */
    int initializeSpeedTable();
    /**
    * Initialize all parameters that may change their value after acquisition setup
    */
    int initializePostSetupParams();
    /**
    * Resizes the buffer used for continuous live acquisition
    */
    int resizeImageBufferContinuous();
    /**
    * Resizes the buffer used for single snaps.
    */
    int resizeImageBufferSingle();
    /**
    * This function reallocates all temporary buffers that might be required
    * for the current acquisition. Such buffers are used when we need to do
    * additional image processing before the image is pushed to MMCore.
    * @throws std::bad_alloc
    * @return Error code if buffer reallocation fails
    */
    int resizeImageProcessingBuffers();

    /**
    * Initiates an acquisition of sequence with a single frame.
    * Called from SnapImage() or the non-circular buffer acquisition thread.
    */
    int acquireFrameSeq();
    /**
    * Called from SnapImage(). Waits until the acquisition of single frame finishes.
    * This method is used for single frame acquisition or by the non-circular buffer
    * acquisition thread.
    */
    int waitForFrameSeq();
    int waitForFrameSeqPolling(const MM::MMTime& timeout);
    int waitForFrameSeqCallbacks(const MM::MMTime& timeout);
    int waitForFrameConPolling(const MM::MMTime& timeout);

    /**
    * Prepares a raw PVCAM frame buffer for use in MM::Core
    * @param [OUT] pOutBuf A pointer to the post processed image buffer. This will point
    *              to one of the internal buffers that were already allocated in reinitProcessingBuffers()
    * @param [IN] pInBuf A raw PVCAM image buffer
    * @param [IN] inBufSz Size of the PVCAM image buffer in bytes
    * @return MM error code
    */
    int postProcessSingleFrame(unsigned char** pOutBuf, unsigned char* pInBuf, size_t inBufSz);

    /**
    * Internally aborts the ongoing acquisition. This method is lock free.
    */
    int abortAcquisitionInternal();

#ifdef PVCAM_SMART_STREAMING_SUPPORTED
    /**
    * Sends the S.M.A.R.T streaming configuration to the camera.
    */
    int sendSmartStreamingToCamera(const std::vector<double>& exposuresMs, int exposureRes);
#endif

    /**
    * This function returns the correct exposure mode to be used in both
    * pl_exp_setup_seq() and pl_exp_setup_cont() functions.
    */
    int16 getPvcamExpMode() const;
    /**
    * This function returns the correct exposure time to be used in both
    * pl_exp_setup_seq() and pl_exp_setup_cont() functions.
    */
    uns32 getPvcamExpTime(double expTimeMs, int expRes) const;
    /**
    * This function gives the bit depth of the image returned by PVCAM for current setup.
    */
    int16 getPvcamBitDepth() const;
    /**
    * This function gives the image format of the image returned by PVCAM for current setup.
    */
    int32 getPvcamImageFormat() const;
    /**
    * This function gives the name of image format of the image returned by PVCAM for current setup.
    */
    const char* getPvcamImageFormatString(int32 value) const;
    /**
    * This function gives the image compression of the image returned by PVCAM for current setup.
    */
    int32 getPvcamImageCompression() const;
    /**
    * This function gives the name image compression of the image returned by PVCAM for current setup.
    */
    const char* getPvcamImageCompressionString(int32 value) const;

    /**
    * This function gives number of channels per pixel based on image format reported by PVCAM.
    */
    unsigned int getPvcamImageChannelsPerPixel() const;
    /**
    * This function gives number of bytes per channel based on image format reported by PVCAM.
    */
    unsigned int getPvcamImageBytesPerChannel() const;

    /**
    * This method is used to estimate how long it might take to read out one frame.
    * The calculation is very inaccurate, it is only used when calculating acquisition timeout.
    */
    unsigned int getEstimatedMaxReadoutTimeMs() const;
    /**
    * Reads current values of all post processing parameters from the camera
    * and stores the values in local array.
    */
    int refreshPostProcValues();
    /**
    * Reverts a single setting that we know had an error
    */
    void revertPostProcValue(long absoluteParamIdx, MM::PropertyBase* pProp);
    /**
    * This function is called right after pl_exp_setup_seq() and pl_exp_setup_cont()
    * After setup is called following parameters become available or may change their values:
    *  PARAM_READOUT_TIME - camera calculated readout time.
    *  PARAM_TEMP_SETPOINT - depends on PARAM_PMODE that is applied during setup.
    *  PARAM_FRAME_BUFFER_SIZE - Frame buffer size depends on setup() arguments.
    * @param frameSize Size of a single frame in bytes as reported by pl_exp_setup() calls.
    */
    int postExpSetupInit(unsigned int frameSize);
    /**
    * Calculates and sets the circular buffer count limits based on frame
    * size and hard-coded limits.
    */
    int updateCircBufRange(unsigned int frameSize);
    /**
    * Selects the correct mask setting for debayering algorithm based on current
    * ROI and sensor physical mask
    * NOTE: The function takes the PVCAM color mode (sensor mask reported by PVCAM)
    * but return the PvDebayer.h interpolation algorithm. These two are basically the
    * same but each group have different values (COLOR_RGGB != CFA_RGGB)
    * @param xRoiPos ROI serial position in sensor coordinates (binning agnostic)
    * @param yRoiPos ROI parallel position in sensor coordinates (binning agnostic)
    * @param pvcamColorMode A color mode as returned by PARAM_COLOR_MODE
    */
    int selectDebayerAlgMask(int xRoiPos, int yRoiPos, int32 colorMask) const;

    /**
    * This function should be called every time the user changes the camera
    * configuration, either by selecting a ROI or changing a Device Property via
    * Device/Property Browser or any other event.
    * This function should validate and apply the new acquisition configuration
    * to the camera, if not accepted the setting should be reverted and error returned.
    * @param forceSetup If true the settings will be sent to camera even without any change.
    */
    int applyAcqConfig(bool forceSetup = false);

private: // Static

#ifdef PVCAM_CALLBACKS_SUPPORTED
    /**
    * Static PVCAM callback handler.
    */
    static void PvcamCallbackEofEx3(FRAME_INFO* pNewFrameInfo, void* pContext);
#endif

private:
    const short     cameraId_;             // 0-based camera ID, used to allow multiple cameras connected
    const std::string deviceName_;         // Name assigned in constructor, returned by GetName

    bool            initialized_;          // Driver initialization status in this class instance
    long            imagesToAcquire_;      // Number of images to acquire
    long            imagesInserted_;       // Current number of images inserted to MMCore buffer
    long            imagesAcquired_;       // Current number of images acquired by the camera
    long            imagesRecovered_;      // Total number of images recovered from missed callback(s)
    short           hPVCAM_;               // Camera handle
    static int      refCount_;             // This class reference counter
    static bool     PVCAM_initialized_;    // Global PVCAM initialization status
    PvDebayer       debayer_;              // debayer processor

    MM::MMTime      startTime_;            // Acquisition start time

    PvCameraModel   cameraModel_;
    char            deviceLabel_[MM::MaxStrLength]; // Cached device label used when inserting metadata

    int             circBufFrameCount_; // number of frames to allocate the buffer for
    bool            circBufFrameRecoveryEnabled_; // True if we perform recovery from lost callbacks

    bool            stopOnOverflow_;       // Stop inserting images to MM buffer if it's full
    bool            snappingSingleFrame_;  // Single frame mode acquisition ongoing
    bool            singleFrameModeReady_; // Single frame mode acquisition prepared
    bool            sequenceModeReady_;    // Continuous acquisition prepared
    bool            callPrepareForAcq_;    // Call PrepareForAcq after {sequence,singleFrame}ModeReady_ is set

    bool            isAcquiring_;

    long            triggerTimeout_;       // Max time to wait for an external trigger

    std::map<int32, std::pair<uns32, uns32>> expTimeResLimits_{}; // [expTimeRes]={min,max}

    friend class    PollingThread;
    PollingThread*  pollingThd_;           // Pointer to the sequencing thread
    friend class    NotificationThread;
    NotificationThread* notificationThd_;  // Frame notification thread
    friend class    AcqThread;
    AcqThread*      acqThd_;               // Non-CB live thread

    StreamWriter*   customDiskWriter_;     // Writer for custom disk streaming feature
    bool            customDiskWriterActive_; // Cached value updated after writer->Start

    /// CAMERA PARAMETERS:
    uns16           camParSize_;           // CCD parallel size
    uns16           camSerSize_;           // CCD serial size

    char            camName_[CAM_NAME_LEN];
    std::string     camChipName_;

    std::vector<std::string>        binningLabels_;
    std::vector<int32>              binningValuesX_;
    std::vector<int32>              binningValuesY_;
    bool                            binningRestricted_;

    double           redScale_;
    double           greenScale_;
    double           blueScale_;

    // Acquisition configuration
    AcqConfig acqCfgCur_; // Current configuration
    AcqConfig acqCfgNew_; // New configuration waiting to be applied

    // Single Snaps and Live mode has each its own buffer. However, depending on
    // the configuration the buffer may need to be further processed before its used by MMCore.

    // PVCAM helper structure for decoding an embedded-metadata-enabled frame buffer
#ifdef PVCAM_METADATA_SUPPORTED
    md_frame*        metaFrameStruct_;
    std::map<uns16, md_ext_item_collection> metaFrameExtData_; // The key is roiNr

    // For metadata serialization, optimization to not allocate the same for each frame
    std::string      metaAllRoisStr_;
    char             metaRoiStr_[1000];
#endif
    // A buffer used for creating a black-filled frame when Centroids or Multi-ROI
    // acquisition is running. Used in both single snap and live mode if needed.
    unsigned char*   metaBlackFilledBuf_;
    size_t           metaBlackFilledBufSz_;
    // A buffer used in setup_seq() only (single snaps mode)
    unsigned char*   singleFrameBufRaw_;
    size_t           singleFrameBufRawSz_;
    // A pointer to the final, post processed image buffer that will be returned
    // in GetImageBuffer() and GetImageBufferAsRGB32(). This is a pointer only that
    // points to either RAW, RGB or Black-Filled buffer.
    unsigned char*   singleFrameBufFinal_;
    // Circular buffer, used in setup_cont() only (live mode)
    PvCircularBuffer circBuf_;
    // Color image buffer. Used in both single snap and live mode if needed.
    ImgBuffer*       rgbImgBuf_;

    Event            eofEvent_;
    MMThreadLock     acqLock_;

#ifdef PVCAM_FRAME_INFO_SUPPORTED
    FRAME_INFO*     pFrameInfo_;           // PVCAM frame metadata
#endif
    int             lastPvFrameNr_;        // The last FrameNr reported by PVCAM

    // All dependant parameters that should be updated after setting new value
    // are listed in the comment after every parameter. For every listed parameter
    // is needed to reset the cache and re-read at least the current value.
    // If there is added '+range', also min/max/inc/def/count values should be updated,
    // and for enum parameters list of valid items enumerated again.
    // If 'all' is added, the parameter can change also remaining attributes,
    // like availability or read-write to/from read-only.
    // The dependencies should be treated recursively.

    // TODO: Convert remaining PvParam pointers to unique_ptr

#ifdef PVCAM_SMART_STREAMING_SUPPORTED
    PvParam<smart_stream_type>* prmSmartStreamingValues_;
    PvParam<rs_bool>* prmSmartStreamingEnabled_;
#endif

    PvEnumParam*      prmTriggerMode_; // Updated after pl_exp_setup_*()
    PvParam<uns16>*   prmExpResIndex_; // Can change: EXP_RES, EXPOSURE_TIME(+range)
    PvEnumParam*      prmExpRes_; // Can change: EXP_RES_INDEX, EXPOSURE_TIME(+range)
    PvParam<ulong64>* prmExposureTime_; // Updated after pl_exp_setup_*()
    PvEnumParam*      prmExposeOutMode_; // Updated after pl_exp_setup_*()

    PvParam<uns16>*   prmClearCycles_;
    PvEnumParam*      prmClearMode_;

    PvEnumParam*      prmReadoutPort_; // Can change: SPDTAB_INDEX(+range)
    PvParam<int16>*   prmSpdTabIndex_; // Can change: PIX_TIME, SPDTAB_NAME, GAIN_INDEX(+range), ADC_OFFSET,
                                       //     COLOR_MODE, IMAGE_FORMAT, IMAGE_COMPRESSION(+range), PP_INDEX
    PvParam<int16>*   prmGainIndex_; // Can change: BIT_DEPTH, GAIN_NAME, SCAN_MODE, GAIN_MULT_FACTOR, TEMP_SETPOINT
    PvParam<uns16>*   prmGainMultFactor_;
    std::unique_ptr<PvParam<int16>>   prmBitDepth_;
    std::unique_ptr<PvEnumParam>      prmImageFormat_;
    std::unique_ptr<PvEnumParam>      prmImageCompression_;
    std::unique_ptr<PvParam<int16>>   prmBitDepthHost_; // Updated after pl_exp_setup_*()
    std::unique_ptr<PvEnumParam>      prmImageFormatHost_; // Updated after pl_exp_setup_*()
    std::unique_ptr<PvEnumParam>      prmImageCompressionHost_; // Updated after pl_exp_setup_*()

    PvEnumParam*      prmColorMode_;
    PvParam<ulong64>* prmFrameBufSize_; // Updated after pl_exp_setup_*()

    PvParam<int16>*   prmTemp_;
    PvParam<int16>*   prmTempSetpoint_;

    PvEnumParam*      prmBinningSer_; // Updated after pl_exp_setup_*()
    PvEnumParam*      prmBinningPar_; // Updated after pl_exp_setup_*()

    PvParam<uns16>*   prmRoiCount_; // Updated after pl_exp_setup_*()
    PvParam<rs_bool>* prmMetadataEnabled_;
    PvParam<rs_bool>* prmMetadataResetTimestamp_;
    PvParam<rs_bool>* prmCentroidsEnabled_;
    PvParam<uns16>*   prmCentroidsRadius_;
    PvParam<uns16>*   prmCentroidsCount_;
    PvEnumParam*      prmCentroidsMode_;
    PvEnumParam*      prmCentroidsBgCount_;
    PvParam<uns32>*   prmCentroidsThreshold_;

    PvEnumParam*      prmFanSpeedSetpoint_; // Can change: TEMP_SETPOINT

    PvEnumParam*      prmTrigTabSignal_; // Can change: LAST_MUXED_SIGNAL(+range)
    PvParam<uns8>*    prmLastMuxedSignal_;

    PvEnumParam*      prmPMode_; // Can change: TEMP_SETPOINT
    PvParam<int16>*   prmAdcOffset_;

    PvEnumParam*      prmScanMode_; // Can change: SCAN_LINE_DELAY(all)/SCAN_WIDTH(all), SCAN_DIRECTION, SCAN_DIRECTION_RESET
    PvEnumParam*      prmScanDirection_;
    PvParam<rs_bool>* prmScanDirectionReset_;
    PvParam<uns16>*   prmScanLineDelay_; // Can change: SCAN_WIDTH, SCAN_LINE_TIME and updated after pl_exp_setup_*()
    PvParam<uns16>*   prmScanWidth_; // Can change: SCAN_LINE_DELAY, SCAN_LINE_TIME and updated after pl_exp_setup_*()
    PvParam<long64>*  prmScanLineTime_; // Updated after pl_exp_setup_*()

    PvParam<uns32>*   prmReadoutTime_; // Available/updated after pl_exp_setup_*()
    PvParam<long64>*  prmClearingTime_; // Available/updated after pl_exp_setup_*()
    PvParam<long64>*  prmPostTriggerDelay_; // Available/updated after pl_exp_setup_*()
    PvParam<long64>*  prmPreTriggerDelay_; // Available/updated after pl_exp_setup_*()

    std::unique_ptr<PvParam<rs_bool>> prmHostFrameSummingEnabled_;
    std::unique_ptr<PvParam<uns32>>   prmHostFrameSummingCount_;
    std::unique_ptr<PvEnumParam>      prmHostFrameSummingFormat_;

    // List of post processing features
    std::vector<PpParam> PostProc_; // PP_PARAM can change: BIT_DEPTH, IMAGE_FORMAT

    // Camera speed table
    //  usage: SpdTabEntry e = camSpdTable_[port][speed];
    std::map<int32, std::map<int16, SpdTabEntry>> camSpdTable_;
    // Reverse speed table to get the speed based on UI selection
    //  usage: SpdTabEntry e = camSpdTableReverse_[port][ui_selected_string];
    std::map<int32, std::map<std::string, SpdTabEntry>> camSpdTableReverse_;
    // Currently selected speed
    SpdTabEntry camCurrentSpeed_;

    // 'Universal' parameters
    std::vector<PvUniversalParam*> universalParams_;
};

#endif //_PVCAMADAPTER_H_
