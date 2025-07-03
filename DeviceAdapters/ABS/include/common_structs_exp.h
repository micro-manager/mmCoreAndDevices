//////////////////////////////////////////////////////////////////////////////
//!
//!
//! \file    common_structs_exp.h
//!
//! \brief    Exported common structs for camera firmware and PC software
//!
//! \author    ABS GmbH Jena (HBau, RG)
//!
//! \date    2006-01-06 -> reorganised
//!
///////////////////////////////////////////////////////////////////////////////
#ifndef _COMMON_STRUCTS_EXPORTED_H_
#define _COMMON_STRUCTS_EXPORTED_H_

#include "datatypes.h"                            // include data type defines

// structure alignment and packing
#if defined(_MSC_VER) || defined(__BORLANDC__)
  // microsoft and compatible specific struct packed pragma
  #pragma pack(push, 1)   //! alignment set to 1 byte
#endif


#ifdef __VISUALDSPVERSION__   // DSP Firmware

  #define _NO_FUNCTION_INCLUDE    // do not include function prototypes
  #include "sensor.h"
  #include "ioports.h"
  #include "memory.h"
//  #include "commprot.h"
  #include "cmdfunctions.h"
  //#include "./JPEG/JPEG.h"
  #define    JPEG_FORMATS             (1)
  #define    TEST_PATTERN_PIXEL       (1)
  #define    MAX_MULTI_ROI            (1)
  #define    DEVICE_FIRMWARE_RANGES   (3)
  #undef _NO_FUNCTION_INCLUDE

#endif // __VISUALDSPVERSION__

#if defined(WIN32) || defined(LINUX)   // Host side

  #define  SENSOR_PIXEL_TYPES          (1)
  #define  SENSOR_GAIN_RANGES          (1)
  #define  SENSOR_EXPOSURE_RANGES      (1)
  #define  SENSOR_DISCRETE_RESOLUTIONS (1)
  #define  SENSOR_LUT_ENTRIES          (1)
  #define  IO_PORTS                    (1)
  #define  MEMORY_REGIONS              (1)
  #define  SENSOR_CLOCK_RANGES         (1)
  #define  TEMPERATURE_SENSORS         (1)
  #define  CAMERA_BUSSES               (1)
  #define  SENSOR_FRAMERATE_RANGES     (1)
  #define  JPEG_FORMATS                (1)
  #define  TEST_PATTERN_PIXEL          (1)
  #define  MAX_MULTI_ROI               (4)
  #define  MAX_TIMES                   (8)
  #define  DEVICE_FIRMWARE_RANGES      (3) // default number of firmware ranges

  //! float to gain value
  #define  F2G_VAL ( _fFloat )      ((u32)  ((_fFloat) * 1000))
  //! gain value to float
  #define  G_VAL2F ( _gain_val )    ((f32)  ((_gain_val) / 1000.0f))

  union U_CAMERA_NOTIFY
  {
    u32   wParam;        //!< window message parameter value

    struct S_CAMERA_NOTIFY
    {
      u08   bUSB_ID;     //!< camera USB-ID
      u08   bPlatformID; //!< camera platform id
      u08   bNotifyCode; //!< camera notification code see #NOTIFY_CODE_CAM_ATTACHED
      u08   bDevNr;      //!< used nDevNr if initialized by this application
    } STRUCT_PACKED;
  };

#endif // defined(WIN32) || defined(LINUX)


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Image-Header
//!@{
// --------------------------------------------------------------------------
// structure for image header
//! \brief  Camera image header
//! (see #CamUSB_GetImage)
//! \note 
//! if you use PIX_JPEG_COMPRESSED than the first u32 of the image data 
//! represent the size of the following jpeg-image-file in memory.
typedef struct {
  u16   wStatus;              //!< unused
  u16   wBlock_id;            //!< unused
  u08   bPacket_format;       //!< format of packet
  u08   bPacket_id_high;      //!< packet id high part
  u16   wPacket_id_low;       //!< packet id low part
  u16   wPayload_ext;         //!< packet payload extension informations
  u16   wPayload_type;        //!< packet payload type

  //! timestamp high u32 see #dwTimestamp_low
  u32   dwTimestamp_high;

  //! \brief timestamp low u32\n
  //! <b>(Millisecond (ms) is the unit of the timestamp which is
  //! compatible to Microsoft time type "time_t". To convert the
  //! timestamp to "time_t" you have to divide the 64Bit
  //! dwTimestamp value by 1000)</b>
  u32   dwTimestamp_low;

  //! pixel type (see \link pixeltypes.h Pixeltypes \endlink)
  u32   dwPixel_type;
  u32   dwSize_x;             //!< horizontal (x) size
  u32   dwSize_y;             //!< vertical (y) size
  i32   dwOffset_x;           //!< horizontal (x) offset
  i32   dwOffset_y;           //!< vertical (y) offset
} STRUCT_PACKED S_IMAGE_HEADER;

typedef struct {
  u16   wExtensions;          //!< count extensions
  u16   wSizeLo;              //!< bytes followed this header including the header size low part
  u16   wSizeHi;              //!< bytes followed this header including the header size high part
} STRUCT_PACKED S_APPENDED_DATA_HDR;

typedef struct {
  u16   wType;                //!< extension type
  u16   wSizeLo;              //!< bytes followed this header including the header size low part
  u16   wSizeHi;              //!< bytes followed this header including the header size high part
} STRUCT_PACKED S_EXTENSION_HDR;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Initialization
//!@{
// --------------------------------------------------------------------------
//! \brief  Camera initialization parameters and device access number
//! (see #CamUSB_InitCameraExS)
//! 
typedef struct
{
  u08   bReserverd[3];        //!< reserved for later user
  u08   nDevNr;               //!< return the device number to address the selected camera
  u32   dwSerialnumber;       //!< if not #NO_SERIAL_NUMBER only a camera with selected serialnumber will be initializied
  u32   dwPlatformID;         //!< if not #CPID_NONE only a camera with selected platform id will be initializied

  //! #FWOPT_AUTOMATIC (start camera firmware if necessary), #FWOPT_KEEP, 
  //! #FWOPT_REBOOT (from FLASH), #FWOPT_START (from file)
  u32   dwFirmwareOptions;
  u08*  pFirmware;          //!< point to camera firmware to be transfered and executed at the camera (ignored if #FO_START not set)
  u32   dwFirmwareSize;     //!< size in bytes of the firmware passed by pFirmware (ignored if #FO_START not set)
  u32   dwReserverd[5];     //!< reserved for later user

} STRUCT_PACKED S_CAMERA_INIT;


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Device Information
//!@{
// --------------------------------------------------------------------------
//! \brief  Camera components and version information
//! (see #CamDev_GetCameraList)
//! Subset of camera information based on #S_CAMERA_VERSION ,
//! #S_CAMERA_STATUS and #S_CAMERA_CFG
typedef struct S_DEVICE_INFO
{
  //! \brief 0 => camera not used by another app.
  //! 1 => camera used by another app.
  u08   bCameraInUse;

  u08   bReserved1[3];      //!< reserved
  u08   bPlatformID;        //!< camera PlatformID #CPID_NONE
  u08   bReserved2[3];      //!< reserved


  //! \brief device serial number\n
  //! (if the function like #CamUSB_GetCameraList returns zero for
  //! this value, the camera is normally in use.)
  u32   dwSerialNumber;
  u08   szDeviceName[32];   //!< short name of device
  u16   wSensorType;        //!< image sensor type, see #ST_MT9T001C
  u16   bReserved3;         //!< reserved

  //! \brief DSP application firmware version (running)\n
  //! (if zero the application firmware isn't active, instead the boot loader should be active)
  u16   wAppVersion;
  u16   bReserved4;         //!< reserved

  u32   dwUserSerialNoLo;   //!< user serial number low part
  u32   dwUserSerialNoHi;   //!< user serial number hi part

  u32   dwCameraType;       //!< type of camera see #ECameraType

} STRUCT_PACKED S_DEVICE_INFO;

//!@}

/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Version Information
//!@{
// --------------------------------------------------------------------------
//! \brief  Camera components and version information
//! (see #CamUSB_GetCameraVersion)
typedef struct
{
  //! \brief size of this structure\n
  //! "dwStructSize" has to be set to the size of this structure!
  //! You should use "sizeof()" operator
  u32   dwStructSize;

  //! \brief device serial number\n
  //! (if the function like #CamUSB_GetCameraList returns zero for
  //! this value, the camera is normally in use.)
  u32   dwSerialNumber;

  u08   szDeviceName[8];      //!< short name of device

  u16   wSensorType;          //!< image sensor type, see #ST_MT9T001C
  u08   bDSPType;             //!< DSP type, see #DSP_NONE
  u08   bFLASHType;           //!< FLASH type, see #FLASH_NONE
  u08   bFPGAType;            //!< FPGA type, see #FPGA_NONE
  u08   bCPLDType;            //!< CPLD type, see #CPLD_NONE
  u08   bUSBType;             //!< USB controller type, see #USB_NONE
  u08   bPlatformID;          //!< camera PlatformID #CPID_NONE
  u08   bHWRevision;          //!< hardware revision
  u08   bReserved[3];         //!< reserved for future use

  //! \brief DSP bootloader firmware version (running)\n
  //! (if zero the bootloader isn't active, instead an application should be active)
  u16   wBootldrVersion;

  //! \brief DSP application firmware version (running)\n
  //! (if zero the application firmware isn't active, instead the bootloader should be active)
  u16   wAppVersion;

  u16   wFPGAVersion;         //!< FPGA configuration version (running)
  u16   wCPLDVersion;         //!< CPLD configuration version (running)

  u16   wBootldrVersionFlash; //!< DSP bootloader firmware version (in flash memory)

  //! \brief DSP application firmware version (in flash memory)\n
  //! (if zero no application is stored at the camera flash memory)
  u16   wAppVersionFlash;

  u16   wFPGAVersionFlash;    //!< FPGA configuration version (in flash memory)
  u16   wCPLDVersionFlash;    //!< CPLD configuration version (in flash memory)

  u32   dwMaxBootLdrSize;     //!< max. size of Bootloader firmware data
  u32   dwMaxAppSize;         //!< max. size of DSP application firmware data
  u32   dwMaxFPGASize;        //!< max. size of FPGA configuration data
  u32   dwMaxCPLDSize;        //!< max. size of CPLD configuration data

  u32    dwSDRAMSize;         //!< camera SDRAM size in Bytes

} STRUCT_PACKED S_CAMERA_VERSION;

typedef struct S_FIRMWARE_VERSION
{
  u32   dwVersion;            //!< current firmware version
  u08   szName[16];           //!< firmware name (zero terminated)
} STRUCT_PACKED S_FIRMWARE_VERSION;

typedef struct S_CAMERA_VERSION_2
{  
  u64   qwSerialNumber;       //!< device serial number
  u08   szDeviceName[32];     //!< short name of device (zero terminated)
  u32   dwSensorID;           //!< image sensor type, see #ST_MT9T001C
  u32   dwPlatformID;         //!< device platform identification #CPID_NONE
  u32   dwTransportID;        //!< device main communication channel #TRANSPORTID_NONE
  u32   dwHWRevision;         //!< hardware revision
  u32   dwReserved[3];        //!< reserved
  u32   dwCountFW;            //!< number of valid firmware versions at sFirmware
  S_FIRMWARE_VERSION sFirmware[ DEVICE_FIRMWARE_RANGES ];
} STRUCT_PACKED S_CAMERA_VERSION_2;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Status Information
//!@{
// --------------------------------------------------------------------------
//! \brief  Camera status information
//! (see #CamUSB_GetCameraStatus)
typedef struct
{
    //!< Error and status flags for camera components see #STATUS_ERROR_USB
  u32    dwDeviceErrorStat;      
  u16    wDSPIntMask;        //!< DSP Interrupt Mask
  u16    wDSPIntPend;        //!< DSP pending interrupts
  u16    wUSBPacketSize;        //!< USB packet size
  u08    bUSBDeviceAddress;      //!< USB device address
  u08    bLastMainRC;        //!< last Return Code from calls in main
} STRUCT_PACKED S_CAMERA_STATUS;

//!@}

/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs:Camera Configuration
//!@{
// --------------------------------------------------------------------------
// structure for Camera Configuration

//! \brief  Camera Configuration
//! Used by #CamUSB_GetCameraCfg. See also #CamUSB_SetCameraCfgValue32 
//! and #CamUSB_SetCameraValue64
typedef struct {
  u32   dwSize;               //!< size of this structure
  u32   dwWriteOptions;       //!< reserved, set to zero
  u32   dwFilterID;           //!< id of installed filter (read only)
  u32   dwGainRed;            //!< boot value of Gain red (read only)
  u32   dwGainGreen;          //!< boot value of Gain green (read only)
  u32   dwGainBlue;           //!< boot value of Gain blue (read only)
  u32   dwReserved1[3];       //!< reserved
  u32   dwOEMID;              //!< OEM - ID
  u32   dwCameraID;           //!< camera user ID
  u32   dwReserved2;          //!< reserved
  u32   dwBootOptions;        //!< camera boot options like #CFG_BO_BOOT_FROM_FLASH
  u32   dwReserved3;          //!< reserved
  u32   dwUserSerialNoLo;     //!< user serial number low part
  u32   dwUserSerialNoHi;     //!< user serial number hi part
  u32   dwReserved4[2];       //!< reserved
  u32   dwMinRecGain;         //!< minimum recommended gain (read only)
  u32   dwRecBlacklevel;      //!< recommended blacklevel (read only)
  u32   dwReserved8[12];      //!< reserved
} STRUCT_PACKED S_CAMERA_CFG;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Device List Element
//!@{
// --------------------------------------------------------------------------
//! \brief Camera Device List Element structure
//! (see #CamUSB_GetCameraList)
typedef struct
{
  S_CAMERA_VERSION  sVersion; //!< camera version structure
  S_CAMERA_STATUS   sStatus;  //!< camera status structure
} STRUCT_PACKED S_CAMERA_LIST;

//!@}

/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Device Extended List Element 
//!@{
// --------------------------------------------------------------------------
//! \brief Camera Device Extended List Element structure
//! (see #CamUSB_GetCameraListEx)
typedef struct
{
  //! \brief 0 => camera not used by another app.
  //! 1 => camera used by another app.
  u08               bCameraInUse;
  u08               bReserved[3]; //!< reserved
  S_CAMERA_VERSION  sVersion;     //!< camera version structure
  S_CAMERA_STATUS   sStatus;      //!< camera status structure
  S_CAMERA_CFG      sCfg;         //!< camera configuration

} STRUCT_PACKED S_CAMERA_LIST_EX;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera function capabilities and parameter structures
//!@{

//! \brief Camera Resolution Data
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_RESOLUTION ROI data
//!
//! see #S_RESOLUTION_CAPS, S_RESOLUTION_RETVALS
typedef struct
{
  i16   wOffsetX;             //!< X offset of ROI (relative to visible area)
  i16   wOffsetY;             //!< Y offset of ROI (relative to visible area)
  u16   wSizeX;               //!< X size (width, columns) of ROI
  u16   wSizeY;               //!< Y size (height, lines) of ROI
  u32   dwSkip;               //!< X- and Y- Skip Settings (see #XY_SKIP_NONE)
  u32   dwBin;                //!< X- and Y- Bin Settings (see #XY_BIN_NONE)
  u08   bKeepExposure;        //!< keep constant exposure (true=yes, false=no)
  u08   bReserved;            //!< reserved
  u16   wResize;              //!< digital image resize (see #XY_RESIZE)

} STRUCT_PACKED S_RESOLUTION_PARAMS, S_RESOLUTION_RETVALS;


//! \brief Camera Resolution Caps\n
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_RESOLUTION
typedef struct
{
  u16   wSensorType;          //!< Sensortype (e.g. #ST_MT9T001C)
  u16   wVisibleSizeX;        //!< viewable pixels (MT9T001C: 2048)
  u16   wVisibleSizeY;        //!< viewable count lines (MT9T001C: 1536)

  i16   wFullOffsetX;         //!< minimum sensor offset X (MT9T001C: -32)
  i16   wFullOffsetY;         //!< minimum sensor offset Y (MT9T001C: -20)
  u16   wFullSizeX;           //!< maximum pixel per line (MT9T001C: 2112)
  u16   wFullSizeY;           //!< maximum count lines (MT9T001C:1568)

  u16   wMaxBPP;              //!< maximum count of bit per pixel
  u32   dwSkipModes;          //!< bit mask of all possible XY-Skip-Settings (see #XY_SKIP_NONE)
  u32   dwBinModes;           //!< bit mask of all possible XY-Bin-Settings  (see #XY_BIN_NONE)

  //! number of discrete resolution settings (0 if continuous setting possible)
  u08   bDiscreteResolutions;

  u08   bReserved[1];         //!< reserved
  u16   wResizeMask;          //!< digital image resize (see #XY_RESIZE)

  //! array of discrete resolutions
  S_RESOLUTION_PARAMS sDiscreteRes[ SENSOR_DISCRETE_RESOLUTIONS ];

} STRUCT_PACKED S_RESOLUTION_CAPS;


//! \brief Camera Resolution Info\n
//! Passed and returned in "pResInfo" by #CamUSB_GetCameraResolutionInfo.
//! The structure contains the resolution which the user want to be set. 
//! In interaction with #CamUSB_GetCameraResolutionInfo the camera ROI constrains 
//! will be applied. The result sResOut and the resulting image dimensions will be
//! return
typedef struct
{
  //! [input] resolution settings which should be checked for
  //! camera resolution constrains
  S_RESOLUTION_PARAMS sResIn;
  //! [input] used pixel type => if zero the the current pixel type will be used
  u32   dwPixelType;

  //! [output] resolution settings with applied camera resolution constrains
  S_RESOLUTION_PARAMS sResOut;
  //! [output] resulting image dimensions based on sResOut
  u16   wImgWidth;          //!< resulting image dimension X
  u16   wImgHeight;         //!< resulting image dimension Y

} STRUCT_PACKED S_RESOLUTION_INFO;

// --------------------------------------------------------------------------


//! \brief Camera Pixel Type Caps\n
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_PIXELTYPE
typedef struct
{
  u32   dwCount;              //!< number of supported pixel types
  //! array of pixel types (see pixeltypes.h)
  u32   dwPixelType[ SENSOR_PIXEL_TYPES ];
} STRUCT_PACKED S_PIXELTYPE_CAPS;


//! \brief Camera Pixel Type Data\n
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_PIXELTYPE
//! \see S_PIXELTYPE_CAPS
typedef struct
{
  u32   dwPixelType;          //!< pixel type (see pixeltypes.h)

} STRUCT_PACKED S_PIXELTYPE_PARAMS, S_PIXELTYPE_RETVALS;

// --------------------------------------------------------------------------
//! \brief Camera Exposure Range\n
//! Used by #S_EXPOSURE_CAPS
typedef struct {
  u32   dwMin;                //!< minimum value
  u32   dwMax;                //!< maximum value
  u32   dwStep;               //!< modify at count of dwStep

} STRUCT_PACKED S_EXPOSURE_RANGE;

//! \brief  Camera Exposure Data\n
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_EXPOSURE \n
//! The exposure unit is in micro seconds (µs).
typedef struct
{
  u32   dwCountRanges;      //!< count exposure ranges

  //! array of exposure range structures
  S_EXPOSURE_RANGE sExposureRange[ SENSOR_EXPOSURE_RANGES ];

} STRUCT_PACKED S_EXPOSURE_CAPS;


//! \brief  Camera Exposure Data
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_EXPOSURE
typedef struct
{
  u32   dwExposure_us;        //!< exposure time in µs

} STRUCT_PACKED S_EXPOSURE_PARAMS, S_EXPOSURE_RETVALS;

// --------------------------------------------------------------------------

//! \brief Camera Gain Range\n
//! Used by #S_GAIN_CAPS
typedef struct
{
  u32   dwMin;                //!< minimum value
  u32   dwMax;                //!< maximum value
  u32   dwStep;               //!< modify at count of dwStep

} STRUCT_PACKED S_GAIN_RANGE;

//! \brief Camera Gain Caps
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_GAIN
//!
//! \remark gains are u32 values in units of 1/1000 factor    \n
//! => GAIN = 13.68      => 13680              \n
//! => GAIN =  3.918237   =>  3918  (237 => is ignored)    \n
//!
//! \remark It is not recommended to set a gain less than the
//! minimum recommended gain value given in wMinRecGain.
typedef struct
{
  u16   wGainChannelMask;     //!< BitMask see "Constants: Gain Channel" (e.g.#GAIN_RED)
  u16   wGainRanges;          //!< Count Gain ranges 1..n
  u08   bGainUnit;            //!< Gain unit see #GAINUNIT_NONE, #GAINUNIT_DB
  u08   bReserved;            //!< reserved
  u16   wMinRecGain;          //!< minimum recommended gain value

  S_GAIN_RANGE sGainRange[ SENSOR_GAIN_RANGES ];  //!< gain ranges

} STRUCT_PACKED S_GAIN_CAPS;

//! \brief max. number of gain channels that can be set in one function call
#define MAX_GAIN_CHANNELS_ONCE    5

//! \brief camera gain data structure
//! used by #CamUSB_SetFunction/#CamUSB_GetFunction
//! => functionID => #FUNC_GAIN
//! \remark the gain flag #GAIN_LOCKED can only be used with #CamUSB_SetGain
typedef struct
{
  u16   wGainChannel;         //!< gain channel to set, more than one channel possible (e.g.#GAIN_RED | #GAIN_BLUE)
  u08   bReserved[2];         //!< reserved, keep 32-bit alignment

  //! \brief gain value for each channel selected by wGainChannel\n
  //! The gain values have to be in the GAIN-Channel bit position order.
  //! Which means for Gain channel mask "GAIN_RED | GAIN_BLUE | GAIN_GREEN"
  //! (0x0001 | 0x0010 | 0x0008) => Bitpos: 0 is red, 4 is blue, 3 is green\n
  //! dwGain[0] => red   \n
  //! dwGain[1] => green \n
  //! dwGain[2] => blue \n
  u32   dwGain[ MAX_GAIN_CHANNELS_ONCE ];
} STRUCT_PACKED S_GAIN_PARAMS, S_GAIN_RETVALS;


// --------------------------------------------------------------------------

//! \brief Camera Look-Up-Table Caps
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_LUT
//!
typedef struct
{
  u16   wLUTBuffers;          //!< number of available LUT buffers (additional to LUT 0)
  u16   wLUTEntries;          //!< number of entries per LUT

} STRUCT_PACKED S_LUT_CAPS;

//! \brief Camera Look-Up-Table Parameters and Return Values\n
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_LUT
typedef struct
{
  u16   wLUTIndex;            //!< index of LUT to set/get
  u16   wLUTDataSize;         //!< size of LUT data in bytes

} STRUCT_PACKED S_LUT_PARAMS, S_LUT_RETVALS;


//! \brief Camera Look-Up-Table Data
//! Passed over parameter "pDataOut" in #CamUSB_SetFunction => functionID => #FUNC_LUT
typedef struct
{
  u16   wLUTData[ SENSOR_LUT_ENTRIES ]; //!< Look-Up-Table Data

} STRUCT_PACKED S_LUT_DATA;


// --------------------------------------------------------------------------

//! \brief  Camera Time Synchronization Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => FUNC_TIMESYNC
//!
typedef struct
{
  //! Mask of supported synchronization modes see #TIMESYNC_POST_TRIG
  u16   wModeMask;
  u16   wReserved;            //!< reserved
  u08   bReserved[60];        //!< reserved for future capabilities

} STRUCT_PACKED S_TIMESYNC_CAPS;


//! \brief  Camera Time Synchronization Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => FUNC_TIMESYNC
typedef struct
{
  //! used synchronization modes see #TIMESYNC_POST_TRIG
  u16   wMode;

  u16   wReserved;            //!< reserved

  //! \brief time value \n
  //! stored as 64Bit value time base 1ms\n
  //! high part of 64Bit time value
  u32   dwTime_high;

  //! \brief low part of 64Bit time value
  u32   dwTime_low;

  u08   bReserved[32];        //!< reserved for future options

} STRUCT_PACKED S_TIMESYNC_PARAMS, S_TIMESYNC_RETVALS;


// --------------------------------------------------------------------------

//! \brief  Camera BlackLevel Correction Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_BLACKLEVEL
//!
typedef struct
{
  i16   wMin;                 //!< min blacklevel (if zero and max also zero only on/off is supported)
  i16   wMax;                 //!< max blacklevel
  i16   wRecBlacklevel;       //!< recommended blacklevel value
  u08   bReserved[58];        //!< reserved for future capabilities

} STRUCT_PACKED S_BLACKLEVEL_CAPS;


//! \brief  Camera BlackLevel Correction Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//! and of MSG return value set for #CamUSB_GetFunction
//! => functionID => FUNC_BLACKLEVEL
typedef struct
{
  u08   bEnable;              //!< enable BlackLevel Correction (0/1)
  u08   bFlags;               //!< Flags => #BLC_LEVELVALUE_IGNORED or #BLC_LEVELVALUE_VALID

  //! \brief Black level value is ignored if the caps min/max value is zero.
  //! If supported the black level value is based on a full scale range
  //! of 12Bit (0..4096). Even if the sensor doesn't support 12Bit.
  i16   wBlackLevel;
  u08   bReserved1[28];       //!< reserved for future options

} STRUCT_PACKED S_BLACKLEVEL_PARAMS, S_BLACKLEVEL_RETVALS;


// --------------------------------------------------------------------------

//! \brief Single Port Capabilities
//! Used by #S_IO_PORT_CAPS
typedef struct
{
  u08   szPortName[8];        //!< port name
  u16   wPortTypeMask;        //!< supported types for this port (e.g. #PORT_TYPE_OUTPUT)
  u16   wPortFeatureMask;     //!< supported features for this port (e.g. #PORT_FEATURE_POL_ACTHIGH)
  u16   wPortStateMask;       //!< supported port states (e.g. #PORT_STATE_SET)
  u16   wMaxDelay;            //!< maximum supported delay (in ms)

} STRUCT_PACKED S_IO_PORT_CAP;


//! \brief Camera I/O Port Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_IO_PORTS
//!
typedef struct
{
  u16   wPorts;               //!< number of available i/o ports
  u08   bReserved[2];         //!< reserved
  //! port capabilities for each i/o port
  S_IO_PORT_CAP sPortCap[ IO_PORTS ];
} STRUCT_PACKED S_IO_PORT_CAPS;


//! \brief Camera I/O Port Parameters and Return Values\n
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_IO_PORTS
typedef struct
{
  u16   wPortIndex;           //!< index of port to set/get
  u16   wPortType;            //!< port type setting (e.g. #PORT_TYPE_OUTPUT)
  u16   wPortFeatures;        //!< port features setting (e.g. #PORT_FEATURE_POL_ACTHIGH)
  u16   wPortState;           //!< port state setting (e.g. #PORT_STATE_SET)
  u32   dwDelay;              //!< port delay setting (µs)

} STRUCT_PACKED S_IO_PORT_PARAMS, S_IO_PORT_RETVALS;


// --------------------------------------------------------------------------

//! \brief  Memory Region Capability
//! Used by #S_MEMORY_CAPS
typedef struct
{
  u16   wTypeMask;            //!< type of memory region (see #MEMORY_TYPE_EEPROM)
  u16   wMemID;               //!< memory id used to access the memory region
  u32   dwSize;               //!< size of region in bytes

} STRUCT_PACKED S_MEMORY_CAP;


//! \brief  Camera Memory Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_MEMORY
//!
typedef struct
{
  u16   wMemories;            //!< number of available memory regions
  u08   bReserved[2];         //!< reserved
  //! capabilities for each memory region
  S_MEMORY_CAP  sMemory[ MEMORY_REGIONS ];
} STRUCT_PACKED S_MEMORY_CAPS;


//! \brief  Camera Memory Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSGreturn value set for #CamUSB_GetFunction
//! => functionID => #FUNC_MEMORY
typedef struct
{
  u16   wMemID;               //!< ID of memory to read/write from/to\n(see #S_MEMORY_CAP::wMemID)
  u08   bReserved[2];         //!< reserved
  u32   dwMemAddress;         //!< address in memory region to begin reading/writing
  u16   wMemDataSize;         //!< size of read/write data in bytes
  u08   bVerify;              //!< verify after write (>0 => on)

} STRUCT_PACKED S_MEMORY_PARAMS, S_MEMORY_RETVALS;


//! \brief  Camera Memory Data
//! Passed over parameter "pDataOut" in #CamUSB_SetFunction => functionID => #FUNC_MEMORY
typedef struct
{
  u08*  pData;                //!< data to read/write from/to memory

} STRUCT_PACKED S_MEMORY_DATA;

// --------------------------------------------------------------------------

//! \brief  Camera Image Flip Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_FLIP
//!
typedef struct
{
  u16   wFlipModeMask;        //!< mask with supported flip settings (see #FLIP_NONE)

} STRUCT_PACKED S_FLIP_CAPS;


//! \brief  Camera Image Flip Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_FLIP
typedef struct
{
  u16   wFlipMode;            //!< desired flip setting (see #FLIP_NONE)

} STRUCT_PACKED S_FLIP_PARAMS, S_FLIP_RETVALS;

// --------------------------------------------------------------------------

//! \brief  Camera Clock ranges
//! Used by #S_CLOCK_CAPS
typedef struct
{
  u32   dwMin;                //!< minimum value in Hz
  u32   dwMax;                //!< maximum value in Hz
  u32   dwStep;               //!< stepping

} STRUCT_PACKED S_CLOCK_RANGE;

//! \brief  Camera Clock Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_CLOCK
//!
typedef struct
{
  u32   dwClockRanges;       //!< count possible clock ranges
  //! array of clock ranges
  S_CLOCK_RANGE sClockRange[ SENSOR_CLOCK_RANGES ];
} STRUCT_PACKED S_CLOCK_CAPS;


//! \brief  Camera Clock Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//! and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_CLOCK
typedef struct
{
  u32   dwClock;              //!< desired clock setting  in Hz

} STRUCT_PACKED S_CLOCK_PARAMS, S_CLOCK_RETVALS;


// --------------------------------------------------------------------------

//! \brief  Temperature Sensor Capability
//! Used by #S_TEMPERATURE_CAPS
typedef struct
{
  u08   szName[8];            //!< sensor name
  u16   wType;                //!< sensor type (see #TEMP_SENS_TYPE_AD7415)
  u16   wUnit;                //!< temperature unit (see #TEMP_SENS_UNIT_C)

} STRUCT_PACKED S_TEMPERATURE_CAP;

//! \brief  Camera Temperature Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps =>
//! functionID => #FUNC_TEMPERATURE
typedef struct
{
  u32   dwSensors;            //!< count possible temperature sensors
  //! array of temperature sensors caps
  S_TEMPERATURE_CAP sSensor[ TEMPERATURE_SENSORS ];
} STRUCT_PACKED S_TEMPERATURE_CAPS;


//! \brief  Camera Temperature Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => #FUNC_TEMPERATURE
typedef struct
{
  //! index of sensor to get (see #S_TEMPERATURE_CAPS::dwSensors)
  u16   wSensorIndex;
  i16   wSensorValue;         //!< temperature sensor value

} STRUCT_PACKED S_TEMPERATURE_PARAMS, S_TEMPERATURE_RETVALS;

// --------------------------------------------------------------------------


//! \brief  Sensor Update lock capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => FUNC_LOCK_UPDATE
//!
typedef struct
{
  u08   bReserved[2];         //!< reserved

} STRUCT_PACKED S_LOCK_UPDATE_CAPS;


//! \brief  Sensor Update lock Parameters and Return Values
//! define of CMD parameter set for #CamUSB_SetFunction
//!  and of MSG return value set for #CamUSB_GetFunction
//! => functionID => FUNC_LOCK_UPDATE
typedef struct
{
  u08   bLock;                //!< lock update (true), enable update (false)
  u08   bReserved;            //!< reserved

} STRUCT_PACKED S_LOCK_UPDATE_PARAMS, S_LOCK_UPDATE_RETVALS;


// --------------------------------------------------------------------------

//! \brief  Camera Bit Shift
//! define of DATA_IN returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_BITSHIFT
typedef struct
{
  u08   bMaxBitShift;         //!< maximum supported bitshift (usually SENSOR_BIT_RESOLUTION - 8)
  u08   bReserved;            //!< reserved

} STRUCT_PACKED S_BITSHIFT_CAPS;

//! \brief  Camera Bit Shift
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_BITSHIFT
typedef struct
{
  u08   bBitShift;            //!< bitshift setting
  u08   bReserved;            //!< reserved

} STRUCT_PACKED S_BITSHIFT_PARAMS, S_BITSHIFT_RETVALS;

// --------------------------------------------------------------------------

//! \brief  Auto Exposure ROI
//! Used by #S_AUTOEXPOSURE_PARAMS and S_AUTOEXPOSURE_RETVALS
typedef struct
{
  u16   wOffsetX;             //!< X offset of ROI
  u16   wOffsetY;             //!< Y offset of ROI
  u16   wSizeX;               //!< X size (width, columns) of ROI
  u16   wSizeY;               //!< Y size (height, lines) of ROI
  u32   dwSkip;               //!< X- and Y- Skip Settings (see #XY_SKIP_NONE)

} STRUCT_PACKED S_AUTOEXPOSURE_ROI;

//! \brief  Camera Auto Exposure Capabilities
//! returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_AUTOEXPOSURE
//! => deprecated
typedef struct
{
  //! 0 => AEC and AGC are only active if both are active \n
  //! 1 => AEC and AGC can be activated separately
  u08   bSeparateAEC_AGC;
  u08   bMaxFrameSkip;        //!< maximum frame skip (is ignored)

  //! maximum target value (brightness), normally 255\n
  u16   wMaxTargetBrightness;
} STRUCT_PACKED S_AUTOEXPOSURE_CAPS;

//! \brief  Camera Auto Exposure Capabilities
//! returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_AUTOEXPOSURE
//!
typedef struct
{
  //! 0   => AEC and AGC are only active if both are active \n
  //! 1   => AEC and AGC can be activated separately
  //! 255 => if this is a S_AUTOEXPOSURE_CAPS2 structure
  //! check S_AUTOEXPOSURE_CAPS2::dwFeatures to see which controls are possible  
  u08   bSeparateAEC_AGC;

  u08   bMaxFrameSkip;        //!< maximum frame skip (is ignored)

  i16   wMaxTargetBrightness; //!< 255 means target mean pixel value is 255 = very bright
  i16   wMinTargetBrightness; //!<   0 means target mean pixel value is 0 = very dark
  u16   wStepTargetBrightness;//!< step size to change wTargetBrightness if used as target brightness

  i16   iMaxBrightnessOffset; //!<   0 means 0dB offset; -1500 mean -1,5dB offset; 10500 mean +10,5dB offset
  i16   iMinBrightnessOffset; //!<   0 means 0dB offset; -1500 mean -1,5dB offset; 10500 mean +10,5dB offset
  u16   wStepBrightnessOffset;//!< step size to change wTargetBrightness if used as target brightness offset
  
  u16   wOptions;             //!< mask of supported options see #AEXP_OPTION_BRIGHNESSOFFSET
  u32   dwFeatures;           //!< mask of supported features see #AEXP_FEATURE_GAIN_EXPOSURE or #AEXP_FEATURE_GAIN_LIMIT

  u32   dwReserved[6];        //!< reserved

} STRUCT_PACKED S_AUTOEXPOSURE_CAPS2;

// \brief  Camera Auto Exposure Parameters and return values \n
// used by #CamUSB_SetFunction/#CamUSB_GetFunction
// => functionID => #FUNC_AUTOEXPOSURE
// \remark if you pass a zero sROI (S_AUTOEXPOSURE_ROI) a default ROI
//  will be used for auto exposure which is relative to the current resolution.
//

//! Camera Auto Exposure Parameters
typedef struct
{
  u08   bAECActive;           //!< AutoExposureControl 0 => inactive 1=> active
  u08   bAGCActive;           //!< AutoGainControl   0 => inactive 1=> active
  u08   bFrameSkip;           //!< (ignored) Frame skip between exposure and gain calculations

  //! \brief (ignored) percentile of luminance relevant pixel
  //! (should be 16% (1%..100%)) this value will be
  //! ignored by some sensors\n
  u08   bBrightnessPercentile;

  //! target brightness value from 0 up to
  //! \link S_RESOLUTION_CAPS::wMaxBPP sensor max bpp \endlink
  //! value (0..255)\n
  //! 255 => means very bright \n
  //! 127 => is default        \n
  //!   0 => means very dark   \n
  //! \n
  //! if S_AUTOEXPOSURE_PARAMS::wOptions bit AEXP_OPTION_BRIGHNESSOFFSET is set\n
  //! wTargetBrightness represents the offset for the internal reference brightness level in dB \n
  //! The minimum/maximum values can be ready over CamUSB_GetFunctionCaps if a S_AUTOEXPOSURE_CAPS2 
  //! structure is passed
  i16   wTargetBrightness;

  //!< (ignored) trigger cycle time in ms (used for auto exposure timeout if SW or HW Triggered capture mode)
  u16   wTriggerCycleTime;

  //! \brief ROI for brightness calculations, if the ROI a zero region
  //! the current resolution will be used for brightness calculations
  //!
  S_AUTOEXPOSURE_ROI  sROI;

  u08   bHysterese;           //!< (ignored) 0 => use internal default; use 1..100%
  u08   bSpeed;               //!< (ignored) 0 => use internal default; use 1..100% of max speed
  u16   wOptions;             //!< see #AEXP_OPTION_BRIGHNESSOFFSET
  u32   dwMinExposure;        //!< min. allowed exposure value for exposure control in µs (0 => default value)

  //! \brief max. allowed exposure value for exposure control in µs (0 => default value)\n
  //! if value != 0 the current framerate settings may be affected, if you switch AutoExposure == off
  //! it is recommended to read out the currently used framerate\n
  u32   dwMaxExposure;

  u32   dwMinGain;            //!< min. allowed gain value for exposure control (0 => default value)
  u32   dwMaxGain;            //!< max. allowed gain value for exposure control (0 => default value)

  u08   bAICActive;           //!< AutoIrisControl 0 => inactive 1=> active
  u08   dwReserved[7];        //!< reserved

} STRUCT_PACKED S_AUTOEXPOSURE_PARAMS, S_AUTOEXPOSURE_RETVALS;

// --------------------------------------------------------------------------

//! \brief  Camera White Balance Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_WHITE_BALANCE
//!
typedef struct
{
  i16    wOffsetX;            //!< minimum offset x
  i16    wOffsetY;            //!< minimum offset y
  u16    wSizeX;              //!< maximum size x (minimum always 4)
  u16    wSizeY;              //!< maximum size y (minimum always 4)
  i16    wMinBalance;         //!< minimum white balance target value
  i16    wMaxBalance;         //!< maximum white balance target value
  
  u32   dwModeMask;           //!< supported white balance modes
  u32   dwOptionMask;         //!< supported white balance options
  u08   bReserverd[40];       //!< reserved

} STRUCT_PACKED S_WHITE_BALANCE_CAPS;

//! \brief  Camera White Balance
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_WHITE_BALANCE
//! \remark if wOffsetX, wOffsetY, wSizeX, wSizeY are set to zero the ROI is ignored, 
//!         the current sensor settings will be used for calculations
typedef struct
{
  i16   wOffsetX;             //!< offset x
  i16   wOffsetY;             //!< offset y
  u16   wSizeX;               //!< size x
  u16   wSizeY;               //!< size y
  i16   wBalance;             //!< white balance target value
  u16   wReserverd;           //!< reserved

  //! white balance mode if set to #WB_MODE_INVALID than #WB_MODE_ONE_PUSH will be used
  u32   dwMode;
  //! white balance option if set to #WB_OPT_INVALID than #WB_OPT_ROI_SENSOR will be used
  u32   dwOption;
  u08   bReserverd[40];       //!< reserved

} STRUCT_PACKED S_WHITE_BALANCE_PARAMS, S_WHITE_BALANCE_RETVALS;

// --------------------------------------------------------------------------

//! \brief  Camera Shading Correction Capabilities
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_SHADING_CORRECTION
//!
typedef struct
{
  u32   dwActions;            //!< supported shading flags see #SHCO_ACTION_ENABLE
  u32   dwFlags;              //!< supported shading flags see #SHCO_FLAG_NONE
  u32   dwReserved[6];        //!< reserved

} STRUCT_PACKED S_SHADING_CORRECTION_CAPS;

//! \brief  Camera Shading Correction
//! used at #CamUSB_GetFunction and #CamUSB_SetFunction => functionID => #FUNC_SHADING_CORRECTION
typedef struct
{
  u32   dwAction;             //!< shading flags see #SHCO_ACTION_ENABLE
  u32   dwFlag;               //!< shading flags see #SHCO_FLAG_NONE
  u32   dwExposure;           //!< valid only for reference creation see #SHCO_FLAG_EXPOSURE
  u32   dwDataSize;           //!< valid only for get operations see #SHCO_FLAG_DATASIZE
  u32   dwExposureDO;         //!< valid only for reference creation see #SHCO_FLAG_EXPOSURE_DO
  u32   dwReserved[3];        //!< reserved

} STRUCT_PACKED S_SHADING_CORRECTION_PARAMS, S_SHADING_CORRECTION_RETVALS;

//! \brief  Camera Shading Correction Data Information
//! used at #CamUSB_GetShadDataInfo
typedef struct
{
  u32   dwType;               //!< version of this header
  u16   wFlip;                //!< flip value used at recording/capture
  u16   wSensorType;          //!< image sensor type, see #ST_MT9T001C
  u32   dwExposure0;          //!< first used exposure during reference creation
  u32   dwExposure1;          //!< second used exposure during reference creation
  u32   dwPixel_type;         //!< pixel type of stored reference data
  u32   dwSize_x;             //!< ROI horizontal (x) size
  u32   dwSize_y;             //!< ROI vertical (y) size
  i32   dwOffset_x;           //!< ROI horizontal (x) offset
  i32   dwOffset_y;           //!< ROI vertical (y) offset
  u08   bBitShift;            //!< bitshift setting
  u08   bTempSensorIndex;     //!< index of sensor to get (see #S_TEMPERATURE_CAPS::dwSensors)
  i16   wTemperature;         //!< temperature sensor value in 1/100°C
  f32   fGain;                //!< green or global gain value (during capture)
  u32   dwReserved[9];        //!< reserved

} STRUCT_PACKED S_SHADING_CORRECTION_DATA_INFO; 


// --------------------------------------------------------------------------
//! \brief  Camera ColorCorrection Capabilities
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_COLOR_CORRECTION
//!
typedef struct
{
  i16   wMin;                 //!< minimum value of a matrix-element
  i16   wMax;                 //!< maximum value of a matrix-element

} STRUCT_PACKED S_COLOR_CORRECTION_CAPS;

//! \brief  Camera Color Correction
//! used as parameter for #CamUSB_SetFunction => functionID => #FUNC_COLOR_CORRECTION
//!
//! \remark wCCMatrix values are from data type Long
//!  values in units of 1/1000 factor => means
//!  => R 11 =  1.41     =>  1410
//!  => B 23 = -0.26     =>  -260
typedef struct
{
  u08   bActive;              //!< 0 => inactive  1 => active

  //! \brief 0 => matrix is ignored,
  //! 1 => matrix will be replaced
  u08   bSetMatrix;

  //! \brief Matrix: R 11 12 13
  //!                G 21 22 23
  //!                B 31 32 33
  i16   wCCMatrix[9];

} STRUCT_PACKED S_COLOR_CORRECTION_PARAMS, S_COLOR_CORRECTION_RETVALS;


// --------------------------------------------------------------------------
//! \brief  Timestamp overlay Capabilities
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_TIMESTAMP_OVERLAY
//!
typedef struct
{
  u32   dwFlagMask;           //!< flags see #OVERLAY_FACTOR_AUTO

} STRUCT_PACKED S_TIMESTAMP_OVERLAY_CAPS;

//! \brief Timestamp overlay
//! used as parameter for #CamUSB_SetFunction => functionID => #FUNC_TIMESTAMP_OVERLAY
typedef struct
{
  u08   bActive;              //!< 0 => inactive  1 => active
  u08   bReserved[3];         //!< reserved
  u32   dwFlag;               //!< flags see #OVERLAY_FACTOR_AUTO

} STRUCT_PACKED S_TIMESTAMP_OVERLAY_PARAMS, S_TIMESTAMP_OVERLAY_RETVALS;

// --------------------------------------------------------------------------
//! \brief Struct: Camera Hue and Saturation (color tone / intensity of color)
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_HUE_SATURATION
typedef struct
{
  i16   wHueMin;              //!< minimum value    (-180° default)
  i16   wHueMax;              //!< maximum value    (+180° default)
  i16   wHueStep;             //!< modify at count of dwStep (1° default)
  i16   wSatMin;              //!< minimum value    (-100 default)
  i16   wSatMax;              //!< maximum value    (+100 default)
  i16   wSatStep;             //!< modify at count of dwStep (1 default)

} STRUCT_PACKED S_HUE_SATURATION_CAPS;

//! \brief Struct: Camera Hue and Saturation (color tone / intensity of color)
//! used as parameter for #CamUSB_SetFunction => functionID => #FUNC_HUE_SATURATION
typedef struct
{
  i16   wHue;                 //!< hue
  i16   wSaturation;          //!< saturation

} STRUCT_PACKED S_HUE_SATURATION_PARAMS, S_HUE_SATURATION_RETVALS;

// --------------------------------------------------------------------------
//! \brief Struct: Camera brightness and contrast
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_BRIGHTNESS_CONTRAST
typedef struct
{
  i16   wContrastMin;         //!< minimum value    (-100 default)
  i16   wContrastMax;         //!< maximum value    (+100 default)
  i16   wContrastStep;        //!< modify at count of dwStep (1 default)
  i16   wBrightnessMin;       //!< minimum value    (-100 default)
  i16   wBrightnessMax;       //!< maximum value    (+100 default)
  i16   wBrightnessStep;      //!< modify at count of dwStep (1 default)

} STRUCT_PACKED S_BRIGHTNESS_CONTRAST_CAPS;

//! \brief  Camera Brightness
//! used as parameter for #CamUSB_SetFunction and #CamUSB_GetFunction =>
//! functionID => #FUNC_BRIGHTNESS_CONTRAST
typedef struct
{
  i16   wBrightness;          //!< brightness
  i16   wContrast;            //!< contrast

} STRUCT_PACKED S_BRIGHTNESS_CONTRAST_PARAMS, S_BRIGHTNESS_CONTRAST_RETVALS;

// --------------------------------------------------------------------------


//! \brief  Camera gamma / value for pixel interpration
//! Returned in "pData" by #CamUSB_GetFunctionCaps => functionID => #FUNC_GAMMA
//! gains are u32 value mean => GAMMA = 1.15  => 1150
//!
typedef struct
{
  u32   dwMin;                //!< minimum value
  u32   dwMax;                //!< maximum value
  u32   dwStep;               //!< modify at count of dwStep

} STRUCT_PACKED S_GAMMA_CAPS;


//! \brief  Camera gamma
//! used as parameter for #CamUSB_SetFunction and #CamUSB_GetFunction =>
//! functionID => #FUNC_GAMMA
typedef struct
{
  u32   dwGamma;              //!< gamma value

} STRUCT_PACKED S_GAMMA_PARAMS, S_GAMMA_RETVALS;

// --------------------------------------------------------------------------


//! \brief  Camera JPEG
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_JPEG
//! define quality factor (1 worst  - 99 best)
//!
typedef struct
{
  u08   bJpegQualityMin;      //!< minimum value
  u08   bJpegQualityMax;      //!< maximum value
  u08   bJpegQualityStep;     //!< modify at count of dwStep
  u08   bReserved[7];         //!< reserved
  u16   wCount;               //!< number of supported JPEG-Formats
  //! array of supported JPEG Formats (see #JPEG_FORMAT1)
  u32   dwJpegFormat[ JPEG_FORMATS ];

} STRUCT_PACKED S_JPEG_CAPS;


//! \brief  Camera JPEG
//! used as parameter for #CamUSB_SetFunction and #CamUSB_GetFunction =>
//! functionID => #FUNC_JPEG
typedef struct
{
  u08   bJpegQuality;         //!< JPEG Quality value (see #S_JPEG_CAPS)
  u08   bReserved[3];         //!< reserved
  u32   dwJpegFormat;         //!< JPEG Format (see #JPEG_FORMAT1)

} STRUCT_PACKED S_JPEG_PARAMS, S_JPEG_RETVALS;


// --------------------------------------------------------------------------


//! \brief  Camera Trigger options
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_TRIGGER_OPTIONS
//!
typedef struct
{
  u32   dwTriggerOptions;     //!< bitfield of available trigger options, see #TRIG_OPTION_NONE
  u32   dwMaxCycleTime_ms;    //!< max. trigger cycle time (in ms), used for timed CCD clear
  u32   dwMaxMinPulseWidth_us;//!< max. value for minimum trigger pulse width settings (in µs)
  u32   dwReserved[5];        //!< reserved

} STRUCT_PACKED S_TRIGGER_OPTION_CAPS;


//! \brief  Camera Trigger options
//! used as parameter for #CamUSB_SetFunction and #CamUSB_GetFunction =>
//! functionID => #FUNC_TRIGGER_OPTIONS
typedef struct
{
  u32   dwTriggerOptions;     //!< bitfield of used trigger options, see #TRIG_OPTION_NONE
  u32   dwCycleTime_ms;       //!< trigger cycle time (in ms), used for timed CCD clear
  u32   dwMinPulseWidth_us;   //!< minimum trigger pulse width settings (in µs)
  u32   dwReserved[5];        //!< reserved

} STRUCT_PACKED S_TRIGGER_OPTION_PARAMS, S_TRIGGER_OPTION_RETVALS;


// --------------------------------------------------------------------------
//! \brief  Camera Framerate Range Data
//! Used by #S_FRAMERATE_CAPS
typedef struct {
  u32   dwMin;                //!< minimum value
  u32   dwMax;                //!< maximum value
  u32   dwStep;               //!< step value

} STRUCT_PACKED S_FRAMERATE_RANGE;


//! \brief  Camera framerate
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_FRAMERATE
//!
//! Framerates are u32 values in units of 1/1000 fps (frames per second) \n
//! => FPS = 25.0      => 25000\n
//! => FPS = 10.5      => 10500\n
//! \n
//! To set the framerate to the default setting the value 0 is used.
//! => FPS = default    => 0                \n
typedef struct
{
  u32   dwCountRanges;        //!< number of framerate ranges

  //! array of framerate range structures
  S_FRAMERATE_RANGE sFramerateRange[ SENSOR_FRAMERATE_RANGES ];
} STRUCT_PACKED S_FRAMERATE_CAPS;


//! \brief  Camera framerate
//! used by #CamUSB_SetFunction/#CamUSB_GetFunction => functionID => #FUNC_FRAMERATE
typedef struct
{
  u32   dwFramerate;          //!< desired (set) or current (get) framerate setting (FPS)
  u32   dwMaxFramerate;       //!< max. framerate setting (FPS)

} STRUCT_PACKED S_FRAMERATE_PARAMS, S_FRAMERATE_RETVALS;

// --------------------------------------------------------------------------
//! \brief  Camera BadPixel Data (stored at camera)
//! Used by #CamUSB_SetFunction => FunctionID #FUNC_BADPIXEL_CORRECTION
typedef struct 
{
  u08   pBadPixData[ 2048 ];  //!< see #mexBlemishDescription

} STRUCT_PACKED S_BADPIXEL_DATA;

//! \brief  Camera BadPixel Data (stored at camera)
//! Used by #CamUSB_SetFunction => FunctionID #FUNC_BADPIXEL_CORRECTION
typedef struct 
{
  u08   pBadPixData[ 8192 ];  //!< see #mexBlemishDescription

} STRUCT_PACKED S_BADPIXEL_DATA2;

//! \brief  Camera BadPixel Data (stored at camera)
//! Used by #CamUSB_SetFunction => FunctionID #FUNC_BADPIXEL_CORRECTION
typedef struct 
{  
  u08   pBadPixData[ 16384 ]; //!< see #mexBlemishDescription

} STRUCT_PACKED S_BADPIXEL_DATA2_5;

//! \brief  Camera BadPixel Data (stored at camera)
//! Used by #CamUSB_SetFunction => FunctionID #FUNC_BADPIXEL_CORRECTION
typedef struct 
{
  u08   pBadPixData[ 32768 ]; //!< see #mexBlemishDescription

} STRUCT_PACKED S_BADPIXEL_DATA3;

//! \brief  Camera BadPixel Data (stored at camera)
//! Used by #CamUSB_SetFunction => FunctionID #FUNC_BADPIXEL_CORRECTION
typedef struct 
{
  u08   pBadPixData[ 64512 ]; //!< see #mexBlemishDescription (63 * 1024))

} STRUCT_PACKED S_BADPIXEL_DATA4;

//! \brief  Camera BadPixel
//! returned by #CamUSB_GetFunctionCaps => functionID => #FUNC_BADPIXEL_CORRECTION
//!
typedef struct
{
  u32   dwOptions;            //!< supported options see #BPC_OPTION_SETDATA
  u32   dwMaxBadPixDataSize;  //!< maximum bad pixel data size, supported by the camera
  u32   dwModes;              //!< supported interpolation modes for bad pixel
  u32   dwReserved[3];        //!< reserved

} STRUCT_PACKED S_BADPIXEL_CAPS;


//! \brief  Camera BadPixel
//! used by #CamUSB_SetFunction/#CamUSB_GetFunction => functionID => #FUNC_BADPIXEL_CORRECTION
typedef struct
{
  u32   dwOption;             //!< BadPixel correction options
  u32   dwState;              //!< switch BadPixel correction on/off  see #BPC_STATE_DISABLED (ignored if not #BPC_OPTION_STATE is set)
  u32   dwMode;               //!< switch the interpolation mode for bad pixel
  u32   dwReserved[3];        //!< reserved

} STRUCT_PACKED S_BADPIXEL_PARAMS, S_BADPIXEL_RETVALS;


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Async Trigger Struct
//!@{
// --------------------------------------------------------------------------
// structure for async trigger struct
//! \brief  async trigger struct
//! (see #CamUSB_TriggerImage)
typedef struct {
  u32   hDevice;              //!< camera device handle

  //! error code of this camera
  i32   iLastError;
} STRUCT_PACKED S_ASYNC_TRIGGER_DEV;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Real Time Clock (RTC) struct
//!@{
// --------------------------------------------------------------------------
//! \brief  RTC \n
//! Used by #CamUSB_GetFunction / #CamUSB_SetFunction => FunctionID => #FUNC_RTC
typedef struct
{
  u16 wYear;                  //!< year  2000..2099
  u16 wMonth;                 //!< months since January - [0,11]
  u16 wDayOfWeek;             //!< days of week (Sunday = 1) - [1,6]
  u16 wDay;                   //!< day of the month - [1,31]
  u16 wHour;                  //!< hours since midnight - [0,23]
  u16 wMinute;                //!< minutes after the hour - [0,59]
  u16 wSecond;                //!< seconds after the minute - [0,59]
  u08 bReserved[2];           //!< reserved

} STRUCT_PACKED S_RTC_PARAMS, S_RTC_RETVALS;

//!@}


//! \cond DOXYGEN_INCLUDE_VDL
/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Video Data Logger (VDL) structs
//!@{
// --------------------------------------------------------------------------
//! \brief  VDL Status \n
//! Used by #CamUSB_GetFunction => FunctionID => #FUNC_VDL_STATUS
typedef struct
{
  u16   wState;               //!< current state (see #VDL_STATE_INACTIVE)
  u08   bReserved[6];         //!< reserved

  u32   dwSDCardState;        //!< SDCard state (see #VDL_SDCARD_DETECT)
  u32   dwSDCardSizeL;        //!< full capacity of SD card in bytes
  u32   dwReserved1;          //!< reserved
  u32   dwSDCardBytesFree;    //!< free capacity of SD card in bytes
  u32   dwReserved2;          //!< reserved

  //! remaining SDCard space in time (msec) to record based on the current camera settings
  u32   dwSDCardRemainingRecTime;
  //! remaining SDRam space in time (msec) to record based on the current camera settings
  u32   dwSDRamRemainingRecTime;
  //! \brief count images currently recorded
  //! only valid at state #VDL_STATE_WR_TO_SDCARD
  u32   dwCountImagesRec;
  //! \brief number of images which should be recorded
  //! only valid at state #VDL_STATE_WR_TO_SDCARD
  u32   dwImagesToRec;

} STRUCT_PACKED S_VDL_STATUS_RETVALS;

// --------------------------------------------------------------------------
//! \brief  VDL Settings Caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_VDL_SETTINGS
typedef struct
{
  u08   bCaptureToMask;       //!< capture to see #VDL_CAPTURE_TO_MEMORY
  u08   bReserved[7];         //!< reserved
  u32   dwMinPreTriggerTime;  //!< min. # time to save before trigger event (in msec.)
  u32   dwMaxPreTriggerTime;  //!< max. # time to save before trigger event (in msec.)
  u32   dwMinPostTriggerTime; //!< min. # time to save after trigger event (in msec.)
  u32   dwMaxPostTriggerTime; //!< max. # time to save after trigger event (in msec.)

} STRUCT_PACKED S_VDL_SETTINGS_CAPS;

//! \brief  VDL Settings \n
//! Used by #CamUSB_GetFunction / #CamUSB_SetFunction => FunctionID => FUNC_VDL_SETTINGS
typedef struct
{
  //! DataLogger Mode global enable/disable (#VDL_STATE_ACTIVE/#VDL_STATE_INACTIVE)
  u08   bState;
  u08   bCaptureTo;           //!< capture to see #VDL_CAPTURE_TO_MEMORY
  u08   bReserved[6];         //!< reserved
  u32   dwPreTriggerTime;     //!< time to save before trigger event (in msec.)
  u32   dwPostTriggerTime;    //!< time to save after trigger event (in msec.)

  //! \brief in case of AVI/JPEG :\n
  //! 0 - use no meta info \n
  //! 1 - use last meta info if present \n
  //! 2 - use meta info following
  u16   wMetaInfo;

  //! \brief in case of wMetaInfo == 2 : size in Byte of Meta info following
  //! see #S_VDL_META_INFO
  u16   wMetaInfoCnt;
} STRUCT_PACKED S_VDL_SETTINGS_PARAMS, S_VDL_SETTINGS_RETVALS;

//! \brief  Meta info \n
//! Used by #CamUSB_SetFunction => FunctionID => FUNC_VDL_SETTINGS
typedef struct
{
  u32   dwCnt;                //!< # of Strings follow
  u08   szDescription[20];    //!< String for image description / Title
  u08   szCommnent[100];      //!< String for comment
  u08   szAuthor[20];         //!< String for author
  u08   szCopyright[10];      //!< String for copyright

} STRUCT_PACKED S_VDL_META_INFO;


// --------------------------------------------------------------------------
//! \brief  VDL Trigger \n
//! Used by #CamUSB_SetFunction => FunctionID => #FUNC_VDL_TRIGGER
typedef struct
{
  u08   bReserved[ 8 ];       //!< reserved

} STRUCT_PACKED S_VDL_TRIGGER_PARAMS;


// --------------------------------------------------------------------------
//! \brief  VDL Directory - Params \n
//! Used by #CamUSB_SetFunction and #CamUSB_GetFunction => FunctionID => #FUNC_VDL_DIR
typedef struct
{
  u08   szBasePath[32];       //!< maximum 2 directories with 12 char per level

  //! operation which should be performed
  u16   wAction;
  u08   dwReserved[6];        //!< reserved

} STRUCT_PACKED S_VDL_DIR_PARAMS;

// --------------------------------------------------------------------------
//! \brief  VDL Directory - Retvals \n
//! Used by #CamUSB_GetFunction => FunctionID => #FUNC_VDL_DIR
typedef struct
{
  u08   szName[16];           //!< returned the name of the item
  u32   dwSize;               //!< returned the size of the item in bytes
  u08   bAttribute;           //!< returned the attributes of the item
  u08   dwReserved[7];        //!< reserved
  u16   wFileTime;            //!< creation Time of File
  u16   wFileDate;            //!< creation Date of File

} STRUCT_PACKED S_VDL_DIR_RETVALS;


// --------------------------------------------------------------------------
//! \brief  VDL File - Params\n
//! Used by #CamUSB_GetFunction => FunctionID => #FUNC_VDL_FILE
typedef struct
{
  u08   szFilePath[48];       //!< path and name of the file

  //! operation which should be performed
  u16   wAction;
  u08   dwReserved[2];        //!< reserved

  //! \brief byte to read from file (for this transfer) \n
  //! (max. VDL_FILE_MAX_BUFSIZE per transfer allowed)
  u32   dwByteToRead;
  //! starting file offset for read / write operation
  u32   dwFileOffset;
} STRUCT_PACKED S_VDL_FILE_PARAMS;

// --------------------------------------------------------------------------
//! \brief  VDL File - Retvals\n
//! Used by #CamUSB_GetFunction => FunctionID => #FUNC_VDL_FILE
typedef struct
{
  u32   dwSize;               //!< returned the size of the item in bytes
  u08   bAttribute;           //!< returned the attributes of the item
  u08   dwReserved[3];        //!< reserved

  //! \brief byte read from file (at this transfer) \n
  //! (max. VDL_FILE_MAX_BUFSIZE per transfer allowed)
  u32   dwByteRead;

  //! file offset for read operation
  u32   dwFileOffset;
} STRUCT_PACKED S_VDL_FILE_RETVALS;

//!@}

//! \endcond DOXYGEN_INCLUDE_VDL


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Edge Enhancement
//!@{
// --------------------------------------------------------------------------
// structure for edge enhancement

//! \brief  edge enhancement caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_EDGE_ENHANCE
typedef struct
{
  u32   dwMin;                //!< min. edge enhancement (0 means deactivated)
  u32   dwMax;                //!< max. strongest edge enhancement
} STRUCT_PACKED S_EDGE_ENHANCE_CAPS;


//! \brief  edge enhancement struct
//! Used by #CamUSB_GetFunction / #CamUSB_SetFunction => FunctionID => #FUNC_EDGE_ENHANCE
typedef struct
{
  //! edge enhancement level (0 means deactivated, 1 is the weakest  up to #S_EDGE_ENHANCE_CAPS::dwMax)
  //! see also #EDGE_ENHANCE_DISABLE
  u32   dwEnhanceLevel;

} STRUCT_PACKED S_EDGE_ENHANCE_PARAMS, S_EDGE_ENHANCE_RETVALS;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera MultiShot Configuration
//!@{
// --------------------------------------------------------------------------
// structure for multishot configuration

//! \brief  multishot configuration caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_MULTISHOT_CFG
typedef struct
{
  u64   qwSupportedFunctionMask;
  u16   wMaxRecords;          //!< max. count of supported records
  u16   wMaxRecordSize;       //!< max. supported record size
  u16   wUsedRecordSize;      //!< used record size, further informations will be ignored by the camera
  u16   wReserved[7];         //!< reserved

} STRUCT_PACKED S_MULTISHOT_CFG_CAPS;

//! \brief camera multishot configuration data
//! Passed over parameter "pDataOut" in #CamUSB_SetFunction => functionID => #FUNC_MULTISHOT_CFG
typedef struct
{
  u64   qwFunctionMask;
  u32   dwExposure_us;        //!< exposure time in µs
  u16   wGainChannel;         //!< gain channel to set, more than one channel possible (e.g.#GAIN_RED | #GAIN_BLUE)
  u08   bReserved[2];         //!< reserved
  u32   dwGain[MAX_GAIN_CHANNELS_ONCE]; //!< gain values
  u32   dwReserved;           //!< reserved

} STRUCT_PACKED S_MULTISHOT_RECORD_DATA;

//! \brief  multishot configuration - params and - retvals \n
//! Used by #CamUSB_SetFunction and #CamUSB_GetFunction => FunctionID => #FUNC_MULTISHOT_CFG\n
typedef struct
{
  //! \brief number of records passed through dataout (0 => disable cfg)\n
  //! The "dataout" size is calculated: wRecords * wRecordSize
  u16   wRecords;
  u16   wRecordSize;          //!< size per record in bytes
  u16   wReserved[8];         //!< reserved

} STRUCT_PACKED S_MULTISHOT_CFG_PARAMS, S_MULTISHOT_CFG_RETVALS;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Multi - ROI Configuration
//!@{
// --------------------------------------------------------------------------

//! \brief  multi ROI configuration - params and - retvals \n
//! Used by #CamUSB_SetFunction and #CamUSB_GetFunction => FunctionID => #FUNC_MULTI_ROI\n
typedef struct
{
  i16   wOffsetX;             //!< X offset of ROI (relative to visible area)
  i16   wOffsetY;             //!< Y offset of ROI (relative to visible area)
  u16   wSizeX;               //!< X size (width, columns) of ROI
  u16   wSizeY;               //!< Y size (height, lines) of ROI

  //! \brief resulting image size based on the ROI, this parameter is only for return values
  //! valid and is affected by pixeltype changes (see #FUNC_PIXELTYPE)\n
  u32   dwImgSize;
} STRUCT_PACKED S_M_ROI;

//! \brief Camera Multi ROI parameter
//! => #FUNC_MULTI_ROI
//!
//! see #S_RESOLUTION_CAPS, S_MULTI_ROI_RETVALS
typedef struct
{
  u32   dwSkip;               //!< X- and Y- Skip Settings (see #XY_SKIP_NONE)
  u32   dwBin;                //!< X- and Y- Bin Settings (see #XY_BIN_NONE)
  u08   bKeepExposure;        //!< keep constant exposure (true=yes, false=no)
  u08   bReserved1[2];        //!< reserved, keep 32-bit alignment
  u08   bUsedROIs;            //!< number of used ROI's (0 => means disabled)
  S_M_ROI   sROI[ MAX_MULTI_ROI ];  //!< array of used ROI's

} STRUCT_PACKED S_MULTI_ROI_PARAMS, S_MULTI_ROI_RETVALS;


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Capture Mode Event Configuration
//!@{
// --------------------------------------------------------------------------
// structure for Capture Mode Event configuration

//! \brief  Capture Mode Event configuration caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_MODE_EVENT_CFG

//! \brief  MODE Event ROI
typedef struct
{
  u16   wOffsetX;             //!< X offset of ROI
  u16   wOffsetY;             //!< Y offset of ROI
  u16   wSizeX;               //!< X size (width, columns) of ROI
  u16   wSizeY;               //!< Y size (height, lines) of ROI

} STRUCT_PACKED S_MODE_EVENT_ROI;


//! \brief MODE Event Caps
typedef struct
{
  u08 bEventTypeMask;         //!< supported Event Types
  u08 bEventOptionMask;       //!< supported Event Options
  u08 bEdgeMask;              //!< supported Edge Types
  u08 bReserved;
  u16 wMaxSizeX;              //!< max X size (width, columns) of ROI
  u16 wMaxSizeY;              //!< max Y size (height, lines) of ROI
  u16 wReserved[6];           //!< reserved

} STRUCT_PACKED S_MODE_EVENT_CFG_CAPS;


//! \brief Capture Mode Event configuration data
//! Passed over parameter "pDataOut" in #CamUSB_SetFunction => functionID => #FUNC_MODE_EVENT_CFG
typedef struct
{
  u08   bEventType;           //!< 1 = Brightness, 2 = motion  (see #EVENT_TYPE_BRIGHTNESS and #EVENT_TYPE_MOTTION)
  u08   bEventOption;         //!< 1 = absolute, 2 = delta (see #EVENT_OPTION_ABSOLUTE)
  u08   bEgde;                //!< 1 = trailing edge , 2 = rising edge , 3 = both edges (see #EVENT_EDGE_FALLING)
  u08   wReserved;            //!< reserved
  u16   wAveragePar;          //!< parameter for moving average, range (0.1 ..0.9)*1000
  u16   wThreshold;           //!< threshold as percentage of full range  1..99% (1% = 10;..99% = 990)

  //! \brief ROI for event detection calculations
  S_MODE_EVENT_ROI  sROI;
  u32   dwReserved[4];        //!< reserved

} STRUCT_PACKED S_MODE_EVENT_CFG_PARAMS, S_MODE_EVENT_CFG_RETVALS;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Capture Mode Time Configuration
//!@{
// --------------------------------------------------------------------------
// structure for Capture Mode Time configuration

//! \brief  Capture Mode Time time\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_MODE_TIME_CFG
typedef struct
{
  u16   wYear;                //!< year  2000..2099
  u08   bMonth;               //!< months since January - [0,11]
  u08   bDay;                 //!< day of the month - [1,31]
  u08   bHour;                //!< hours since midnight - [0,23]
  u08   bMinute;              //!< minutes after the hour - [0,59]
  u08   bSecond;              //!< seconds after the minute - [0,59]
  u08   bReserved;            //!< reserved

} STRUCT_PACKED S_DATETIME;

//! \brief  Capture Mode Time configuration caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_MODE_TIME_CFG
typedef struct
{
  u32   dwIntervalMin;        //!< minimum value
  u32   dwIntervalMax;        //!< maximum value
  u32   dwIntervalStep;       //!< modify at count of dwStep
  u32   dwFlagMask;           //!< supported flag mask
  S_DATETIME  sDateTimeMax;   //!< maximum time value

} STRUCT_PACKED S_MODE_TIME_CFG_CAPS;

//! \brief Capture Mode Event configuration data
//! Passed over parameter "pDataOut" in #CamUSB_SetFunction => functionID => #FUNC_MODE_TIME_CFG
typedef struct
{
  u32   dwInterval;           //!< interval time in ms between an image capture
  u32   dwFlags;              //!< not used
  u32   dwValidDateTimes;     //!< count valid time values, at which an image should be captured

  //! structure with dates for capture an image
  S_DATETIME sDateTime[ MAX_TIMES ];
} STRUCT_PACKED S_MODE_TIME_CFG_PARAMS, S_MODE_TIME_CFG_RETVALS;


//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera Event Notification Structures
//!@{
// --------------------------------------------------------------------------
// structure for SetEventNotification / GetEventNotification

//! \brief Event Notification Data
//! Passed over parameter "pEventNotify" in #CamUSB_SetEventNotification and #CamUSB_GetEventNotification
typedef struct
{
  u16   wEventID;             //!< see #EVENTID_START_OF_TRANSFER, #EVENTID_END_OF_TRANSFER
  u16   wNotificationType;    //!< see #NOTIFY_TYPE_EVENT and #NOTIFY_TYPE_MESSAGE

  //! used as Event-Handle  for Event-Notifications
  //! used as Window-Handle for Message-Notifications
  //! if this value is 0 the Notification will be cleared
  #ifdef _WIN64
    u32   dwReserved0;        // padding
    u64   hHandle;
    u32   dwReserved1;        // padding
  #else
    u32   hHandle;
  #endif

  u32   dwMessage;            //!< used as windows message for SendMessage (valid with #NOTIFY_TYPE_MESSAGE)
  u32   lParam;               //!< passed as LPARAM to SendMessage (valid with #NOTIFY_TYPE_MESSAGE)
  u32   wParam;               //!< passed as WPARAM to SendMessage (valid with #NOTIFY_TYPE_MESSAGE)

} STRUCT_PACKED S_EVENT_NOTIFICATION;

   
//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera function image sensor cooling
//!@{
// --------------------------------------------------------------------------
// structure for image sensor cooling

//! \brief  Image sensor cooling caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_COOLING

//! \brief cooling caps
typedef struct
{
  u08   bCoolingModeMask;     //!< supported cooling modes #CM_TEC_OFF, #CM_TEC_AUTOMATIC or #CM_TEC_MANUAL
  u08   bCoolingLevelMin;     //!< minimum supported cooling level in percent (only for #CM_TEC_AUTOMATIC)
  u08   bCoolingLevelMax;     //!< maximum supported cooling level in percent (only for #CM_TEC_AUTOMATIC)
  u08   bCoolingLevelStep;    //!< cooling level step size in percent, e.g. 1 corresponds 1% and 100 => 100% (only for #CM_TEC_AUTOMATIC)
  i16   wTargetTempMin;       //!< minimum supported target temperature value for automatic cooling
  i16   wTargetTempMax;       //!< maximum supported target temperature value for automatic cooling
  u16   wTargetTempStep;      //!< possible temperature step size (e.g. 10 corresponds 1.0°C and 5 => 0.5°C and -124 = -12.4°C)
  u16   wMinPeltierVoltage;   //!< obsolate don't use (minimum peltier voltage e.g. 600 mean 0.6 V)
  u16   wMaxPeltierVoltage;   //!< obsolate don't use (maximum peltier voltage 9400 mean 9.4 V)

  u08   bFanModeMask;         //!< supported Fan modes #CM_FAN_OFF, #CM_FAN_AUTOMATIC or #CM_FAN_MANUAL
  u08   bFanLevelMin;         //!< minimum supported Fan level in percent (only for #CM_FAN_MANUAL)
  u08   bFanLevelMax;         //!< maximum supported Fan level in percent (only for #CM_FAN_MANUAL)
  u08   bFanLevelStep;        //!< Fan level step size in percent, e.g. 1 means 1% and 100 => 100% (only for #CM_FAN_MANUAL)
  i16   wFanThreshTempMin;    //!< minimum supported threshold temperature in 0.1°C, e.g. 10 => 1.0°C (only for #CM_FAN_AUTOMATIC)
  i16   wFanThreshTempMax;    //!< maximum supported threshold temperature in 0.1°C (only for #CM_FAN_AUTOMATIC)
  i16   wFanThreshTempStep;   //!< minimum supported threshold temperature in 0.1°C(only for #CM_FAN_AUTOMATIC)
  u08   bReserved[40];        //!< reserved

} STRUCT_PACKED S_COOLING_CAPS;


//! \brief Image sensor cooling parameter / return values
//! Passed as "pCmd" or "pMsg" parameter in #CamUSB_SetFunction / #CamUSB_GetFunction => functionID => #FUNC_COOLING
typedef struct
{
  u08   bCoolingMode;         //!< #CM_TEC_OFF, #CM_TEC_AUTOMATIC or #CM_TEC_MANUAL
  u08   bReserved1[3];        //!< reserved
  i16   wTargetTemp;          //!< in 1/10 °C (10 => 1.0°C) only used with cooling mode #CM_TEC_AUTOMATIC
  u08   bReserved2[2];        //!< reserved
  u08   bCoolingLevel;        //!< in percent (%) (15 => 15 %) only used with cooling mode #CM_TEC_MANUAL
  
  u08   bReserved3[3];        //!< reserved
  u08   bFanMode;             //!< current Fan mode (#CM_FAN_OFF, #CM_FAN_AUTOMATIC or #CM_FAN_MANUAL), must be #CM_FAN_AUTOMATIC if #CM_TEC_AUTOMATIC is active)
  u08   bReserved4[3];        //!< reserved
  i16   wFanThreshTemp;       //!< current threshold temperature in 0.1°C (only used with fan mode #CM_FAN_AUTOMATIC)
  u08   bReserved5[2];        //!< reserved
  u08   bFanLevel;            //!< current Fan level in percent (only used with fan mode #CM_FAN_MANUAL)
  u08   bReserved[43];        //!< reserved

} STRUCT_PACKED S_COOLING_PARAMS, S_COOLING_RETVALS;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera function image color mapping (mono to color)
//!@{
// --------------------------------------------------------------------------
// structure for image color mapping (mono to color)

//! \brief  Image color mapping caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_COLOR_MAPPING

//! \brief color mapping caps
typedef struct
{
  u32   dwMode;               //!< supported color mapping modes #COMA_MODE_COLORMAPPING or #COMA_MODE_BITMAPPING
  u32   dwMaxColorMappingEntries; //!< max. number of supported color mapping entries #S_COLOR_MAPPING_DATA_CM
} STRUCT_PACKED S_COLOR_MAPPING_CAPS;

//! \brief color mapping data mode "color mapping"
//! Used by #CamUSB_GetFunction / #CamUSB_SetFunction => FunctionID => #FUNC_COLOR_MAPPING, 
//! if the mode COMA_MODE_COLORMAPPING and the flag COMA_FLAG_DATA is set at the command structure,
//! as data array passed in pData.
typedef struct
{
  f64   fGrayValue;           //!< gray value
  f64   fRed;                 //!< red value
  f64   fGreen;               //!< green value
  f64   fBlue;                //!< blue value

} STRUCT_PACKED S_COLOR_MAPPING_DATA_CM;

//! \brief color mapping data mode "bit mapping"
//! Used by #CamUSB_GetFunction / #CamUSB_SetFunction => FunctionID => #FUNC_COLOR_MAPPING, 
//! if the mode COMA_MODE_BITMAPPING and the flag COMA_FLAG_DATA is set at the command structure,
//! as data passed in pData.
typedef struct
{
  u16   wRed[16];             //!< new values for bit0 - bit15 at red channel
  u16   wGreen[16];           //!< new values for bit0 - bit15 at green channel
  u16   wBlue[16];            //!< new values for bit0 - bit15 at blue channel

} STRUCT_PACKED S_COLOR_MAPPING_DATA_BM;

//! \brief Image color mapping parameter
//! Passed as "pCmd" or "pMsg" parameter in #CamUSB_SetFunction / #CamUSB_GetFunction => functionID => #FUNC_COLOR_MAPPING
typedef struct
{
  //! color mapping mode #COMA_MODE_COLORMAPPING, #COMA_MODE_BITMAPPING, #COMA_MODE_DISABLED
  u32   dwMode;
  //! number of S_COLOR_MAPPING_DATA_CM entries passed over data pipe,
  //! only used with mode #COMA_MODE_COLORMAPPING
  u32   dwColorMappingEntries;
} STRUCT_PACKED S_COLOR_MAPPING_PARAMS, S_COLOR_MAPPING_RETVALS;

//!@}


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera function image rotate / remap
//!@{
// --------------------------------------------------------------------------
// structure for image rotation / remapping

//! \brief  Image rotation / remapping caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_ROTATE_REMAP

//! rotation remap capabilities
typedef struct
{
  u32   dwModes;              //!< supported operation modes #ROTREM_MODE_OFF
  u32   dwFlags;              //!< supported flags see #ROTREM_FLAG_NONE
  u32   dwInterpolations;     //!< supported interpolations see #ROTREM_INTER_NN
  u32   dwReserved[13];       //!< reserved

} STRUCT_PACKED S_ROTATE_REMAP_CAPS;


//! \brief Image rotation / remapping parameter
//! Passed as "pCmd" or "pMsg" parameter in #CamUSB_SetFunction / #CamUSB_GetFunction => functionID => #FUNC_ROTATE_REMAP
//! 
//! rotation remap params / retvals
typedef struct
{
  u32   dwMode;               //!< operation modes #ROTREM_MODE_OFF
  u32   dwFlag;               //!< used flag see #ROTREM_FLAG_NONE
  u32   dwInterpolation;      //!< used interpolations see #ROTREM_INTER_NN
  u32   dwDataSize;           //!< size of passed remap data (only used with #ROTREM_FLAG_DATASIZE)
  f32   fAngle;               //!< rotation angle (positive => rotate right)
  f32   fOriginX;             //!< x position of rotation origin (only used with #ROTREM_MODE_ROTATE_XY)
  f32   fOriginY;             //!< y position of rotation origin (only used with #ROTREM_MODE_ROTATE_XY)
  f32   fOffsetX;             //!< left image position of the returned image after rotation (only used with #ROTREM_FLAG_ROI)
  f32   fOffsetY;             //!< top image position of the returned image after rotation (only used with #ROTREM_FLAG_ROI)
  u32   dwSizeX;              //!< image width of the returned image after rotation (only used with #ROTREM_FLAG_ROI)
  u32   dwSizeY;              //!< image height of the returned image after rotation (only used with #ROTREM_FLAG_ROI)
  u32   dwReserved[6];        //!< reserved

} STRUCT_PACKED S_ROTATE_REMAP_PARAMS, S_ROTATE_REMAP_RETVALS;


/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Common range
//!@{

// --------------------------------------------------------------------------
//! \brief u32 value range
typedef struct {
  u32   dwMin;                //!< minimum value
  u32   dwMax;                //!< maximum value
  u32   dwStep;               //!< modify at count of dwStep

} STRUCT_PACKED S_RANGE_U32;

//!@}

/////////////////////////////////////////////////////////////////////////////
//! \name Typedefs: Camera function seams correction
//!@{
// --------------------------------------------------------------------------
// structure for image seams correction => compansation of brightnes and gain
// differences between the image sensor readout quadrants

#define SENSOR_SEAMS_GAIN_RANGES 1
#define SENSOR_SEAMS_OFFSET_RANGES 1

//! \brief Offset capabilities \n
typedef struct
{
  i32   dwMin;
  i32   dwMax;
  i32   dwStep;
} STRUCT_PACKED S_OFFSET_RANGE;

//! \brief  Seams correction caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_SEAMS_CORRECTION
//!
typedef struct
{
  u16                     wQuadrants;                                   //!< count supported qudrants
  u08                     bGainUnit;                                    //!< Gain unit see #GAINUNIT_NONE, #GAINUNIT_DB
  u08                     bReserved[1];                                 //!< reserved
  u32                     dwDefaultGain;
  i32                     iDefaultOffset;
  S_GAIN_RANGE            sGainRanges[ SENSOR_SEAMS_GAIN_RANGES ];      //!< gain ranges
  S_OFFSET_RANGE          sOffsetRanges[ SENSOR_SEAMS_OFFSET_RANGES ];  //!< offset ranges
} STRUCT_PACKED S_SEAMS_CORRECTION_CAPS;

//! \brief  Seams correction caps\n
//! Used by #CamUSB_GetFunctionCaps => FunctionID => #FUNC_SCENES_CORRECTIONEN
typedef struct
{
  u16   wQuadrant;        //!< number/index of addressed quadrant to setup correction
  u08   bReserved[2];     //!< reserved
  i32   iOffset;          //!< offset
  u32   dwGain;           //!< gain factor (1/1000) based of support unit see caps
} STRUCT_PACKED S_SEAMS_CORRECTION_PARAMS, S_SEAMS_CORRECTION_RETVALS;

//!@}

// structure alignment and packing
#if defined(_MSC_VER) || defined(__BORLANDC__)
  // restore default alignment
  #pragma pack(pop) 
#endif

#endif //_COMMON_STRUCTS_EXPORTED_H_
