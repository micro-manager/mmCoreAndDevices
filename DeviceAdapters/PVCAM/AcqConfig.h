#ifndef _ACQCONFIG_H_
#define _ACQCONFIG_H_

#include "PvRoiCollection.h"

#include <map>
#include <string>
#include <vector>

enum AcqType
{
    AcqType_Snap = 0,
    AcqType_Live
};

/**
* This class holds the camera and acquisition configuration. The idea is that
* if user changes a property during live mode the configuration is stored in
* "future" configuration class. Once the live mode is restarted the "future"
* configuration is compared to "current" configuration and the camera, buffers
* and other acquisition variables are set accordingly.
* This approach resolves the problem that comes from how MM handles property
* changes during live mode:
* 1) The OnProperty() handler is called
* 2) StopSequenceAcquisition() is called
* 3) StartSequenceAcquisition() is called
* Because the OnProperty() is called during live acquisition some intermediate
* variables cannot be updated right away - for example binning or ROI because
* these are still used in ongoing acquisition. Instead, we need to remember the
* "new" or "future" state that must be applied once StartSequenceAcquisition()
* is called.
* Previously we had many variables like: newColorMode, currentColorMode, newBinning,
* currentBinning, etc. These are, or should be all transferred to this class.
*/
class AcqConfig
{
public:
    AcqConfig();

public:
    /**
    * All members are public to simplify the usage.
    * There is nothing to be done in setters and getters.
    * Values set here are either invalid or represent the default camera state.
    * Each value can be overridden during camera initialization.
    */

    /**
    * Micro-Manager exposure time in milli-seconds. This is later converted
    * to PVCAM exposure time based on current exposure resolution.
    */
    double ExposureMs{ 10 };
    /**
    * Current PVCAM exposure resolution - EXP_RES_ONE_MILLISEC, MICROSEC, etc.
    */
    int    ExposureRes{ EXP_RES_ONE_MILLISEC };
    /**
    * Type of acquisition the camera should be prepared for
    */
    AcqType AcquisitionType{ AcqType_Snap };
    /**
    * Embedded frame metadata enabled or disabled.
    */
    bool FrameMetadataEnabled{ false };
    /**
    * Resets the camera-generated metadata timestamp.
    */
    bool FrameMetadataResetTimestamp{ false };
    /**
    * Centroids - camera selected ROIs enabled or disabled.
    */
    bool CentroidsEnabled{ false };
    /**
    * Centroids radius.
    */
    int CentroidsRadius{ 0 };
    /**
    * Number of centroids to acquire.
    */
    int CentroidsCount{ 0 };
    /**
    * Centroids mode.
    */
    int CentroidsMode{ PL_CENTROIDS_MODE_LOCATE };
    /**
    * Number of for dynamic background removal. It is enumeration without any enum type defined.
    */
    int CentroidsBgCount{ 0 };
    /**
    * Threshold multiplier for specific object detection mode. A raw value from camera.
    * A fixed-point real number in Q8.4 format.
    */
    int CentroidsThreshold{ 100 };
    /**
    * Total number of "output" ROIs for the current acquisition. This could be either
    * the Centroids Count or number of user defined ROIs (1 or more if supported)
    */
    int RoiCount{ 1 };
    /**
    * Selected fan speed.
    */
    int FanSpeedSetpoint{ FAN_SPEED_HIGH };
    /**
    * Regions of interest. Array of input ROIs that will be sent to the camera.
    */
    PvRoiCollection Rois{};
    /**
    * Number of sensor clearing cycles.
    */
    int ClearCycles{ 2 };
    /**
    * Selected clearing mode. PARAM_CLEAR_MODE values.
    */
    int ClearMode{ CLEAR_PRE_EXPOSURE };
    /**
    * Color on or off.
    */
    bool ColorProcessingEnabled{ false };
    /**
    * Selected mask used for debayering algorithm (must correspond to CFA masks defined
    * in PvDebayer.h - CFA_RGGB, CFA_GRBG, etc.)
    */
    int DebayerAlgMask{ 0 };
    /**
    * Enables / disables the automatic selection of sensor mask for debayering algorithm.
    * The mask changes with odd ROI and may change with different port/speed combination.
    */
    bool DebayerAlgMaskAuto{ false };
    /**
    * This must correspond to defines in PvDebayer.h (ALG_REPLICATION, ALG_BILINEAR, etc)
    */
    int DebayerAlgInterpolation{ 0 };
    /**
    * A map of trigger signals and their muxing settings.
    *  key = PARAM_TRIGTAB_SIGNAL value
    *  val = PARAM_LAST_MUXED_SIGNAL value
    * Example:
    *  ExposeOutSignal: 4
    *  ReadoutSignal: 2
    */
    std::map<int, int> TrigTabLastMuxMap{};
    /**
    * Current PMode value
    */
    int PMode{ PMODE_NORMAL };
    /**
    * Current ADC offset
    */
    int AdcOffset{ 0 };
    /**
    * Scan Mode
    */
    int ScanMode{ PL_SCAN_MODE_AUTO };
    /**
    * Scan Direction
    */
    int ScanDirection{ PL_SCAN_DIRECTION_DOWN };
    /**
    * Scan Direction Reset State (ON/OFF)
    */
    bool ScanDirectionReset{ true };
    /**
    * Scan Line Delay
    */
    int ScanLineDelay{ 0 };
    /**
    * Scan Line Width
    */
    int ScanWidth{ 0 };
    /**
    * Current port ID.
    */
    int PortId{ 0 };
    /**
    * Current speed index.
    */
    int SpeedIndex{ 0 };
    /**
    * Current gain number.
    */
    int GainNum{ 0 };
    /**
    * Current image format.
    */
    int ImageFormat{ PL_IMAGE_FORMAT_MONO16 };
    /**
    * Current image compression.
    */
    int ImageCompression{ PL_IMAGE_COMPRESSION_NONE };
    /**
    * Whether to use circular buffer for live acquisition or not
    */
    bool CircBufEnabled{ true };
    /**
    * Whether to use adjust the circular size automatically based on acquisition configuration
    */
    bool CircBufSizeAuto{ true };
    /**
    * True if PVCAM callbacks are active, false to use polling
    */
    bool CallbacksEnabled{ true };
    /**
    * Enables or disables custom streaming to disk.
    * Please note that this streaming is enabled for continuous acquisition only.
    * The streaming to disk is fully controlled by PVCAM adapter and should only
    * be used in cases where standard MDA is unable to keep up with camera speed
    * when storing the data, esp. at high data rates, greater than 2GB/s.
    */
    bool DiskStreamingEnabled{ false };
    /**
    * The path where files with raw data will be stored.
    */
    std::string DiskStreamingPath{};
    /**
    * Ratio of images forwarded to the core.
    * The value 1 means all frames are sent, value 2 means every second frame is sent, etc.
    * Values higher than 1 result in frame rate counter showing a lower value
    * than the actual acquisition rate.
    */
    int DiskStreamingCoreSkipRatio{ 100 };
    /**
    * Enables or disables the S.M.A.R.T streaming. Please note that the S.M.A.R.T streaming
    * mode is enabled for continuous acquisition only. See the "Active" variable.
    */
    bool SmartStreamingEnabled{ false };
    /**
    * Controls whether the S.M.A.R.T streaming is actually active or not. In Single snap mode
    * we always temporarily disable the S.M.A.R.T streaming and use the exposure value from
    * the main GUI. (S.M.A.R.T streaming does not have any effect in single snaps)
    */
    bool SmartStreamingActive{ false };
    /**
    * Exposure values for S.M.A.R.T streaming in milliseconds
    */
    std::vector<double> SmartStreamingExposuresMs{ 10, 20, 30, 40 };
    /**
    * SW frame summing in PVCAM - disabled by default.
    */
    bool HostFrameSummingEnabled{ false };
    /**
    * SW frame summing in PVCAM - number of frames to sum.
    * It should be unsigned int type, but UI property supports long, double or
    * string only. Using long type here covers all values on Linux and Unix-like
    * systems, while for Windows (where long is 32-bit only) the max. value
    * should be limited. However, it is questionable weather summing over
    * 2 billion frames is real case or not.
    */
    long HostFrameSummingCount{ 5 };
    /**
    * SW frame summing in PVCAM - image format of output frame.
    */
    int HostFrameSummingFormat{ PL_FRAME_SUMMING_FORMAT_16_BIT };
    /**
    * Set to true whenever the PAPRAM_PP_PARAM value is set.
    * It behaves like a trigger to initiate acq. setup because e.g. HW frame
    * summing can change bit depth and pixel size (16-bit, 32-bit, ...).
    */
    bool PostProcParamSet{ false };
};

#endif
