//-----------------------------------------------------------------//
// Name        | sc2_sdkstructures.h         | Type: ( ) source    //
//-------------------------------------------|       (*) header    //
// Project     | SC2                         |       ( ) others    //
//-----------------------------------------------------------------//
// Purpose     | SC2 - Structure defines                           //
//-----------------------------------------------------------------//
// Author      |  FRE, Excelitas PCO GmbH                          //
//-----------------------------------------------------------------//
// Notes       | Covers following groups of structures and the     //
//             | famous camera descriptor:                         //
//             | 1. General control group.                         //
//             | 2. Sensor control group.                          //
//             | 3. Timing control group.                          //
//             | 4. Storage control group.                         //
//             | 5. Recording control group.                       //
//             | 6. Image read group.                              //
//             | Each data entry in the structure will be defined  //
//             | in the way as they are defined on the Firmware.   //
//             |                                                   //
//             | See pco.camera SDK manual for further information.//
//-----------------------------------------------------------------//
// Attention!! | Attention!! If these structures are released to   //
//             | market in any form, the position of each data     //
//             | entry must not be moved any longer! Enhancements  //
//             | can be done by exploiting the dummy entries and   //
//             | dummy fields.                                     //
//-----------------------------------------------------------------//
// (c) 2021 Excelitas PCO GmbH * Donaupark 11 *                    //
// D-93309      Kelheim / Germany * Phone: +49 (0)9441 / 2005-0 *  //
// Fax: +49 (0)9441 / 2005-20 * Email: pco@excelitas.com           //
//-----------------------------------------------------------------//


// The wZZAlignDummies are only used in order to reflect the compiler output.
// Default setting of the MS-VC-compiler is 8 byte alignment!!
// Despite the default setting of 8 byte the MS compiler falls back to the biggest member, e.g.
// in case the biggest member in a struct is a DWORD, all members will be aligned to 4 bytes and
// not to default of 8.

#if !defined  SC2_STRUCTURES_H
#define SC2_STRUCTURES_H

// Defines: 
// WORD: 16bit unsigned
// SHORT: 16bit signed
// DWORD: 32bit unsigned

#define PCO_STRUCTREV      102         // set this value to wStructRev

#define PCO_BUFCNT 16                   // see PCO_API struct
#define PCO_MAXDELEXPTABLE 16          // see PCO_Timing struct
#define PCO_RAMSEGCNT 4                // see PCO_Storage struct
#define PCO_MAXVERSIONHW   10
#define PCO_MAXVERSIONFW   10


#define PCO_ARM_COMMAND_TIMEOUT 10000
#define PCO_HPX_COMMAND_TIMEOUT 10000
#define PCO_COMMAND_TIMEOUT       400

// SDK-Dll internal defines (different to interface type in sc2_defs.h!!!
// In case you're going to enumerate interface types, please refer to sc2_defs.h.
#define PCO_INTERFACE_FW      1         // Firewire interface
#define PCO_INTERFACE_CL_MTX  2         // Cameralink Matrox Solios / Helios
#define PCO_INTERFACE_GENICAM 3         // Genicam wrapper interface
#define PCO_INTERFACE_CL_NAT  4         // Cameralink National Instruments
#define PCO_INTERFACE_GIGE    5         // Gigabit Ethernet
#define PCO_INTERFACE_USB     6         // USB 2.0
#define PCO_INTERFACE_CL_ME4  7         // Cameralink Silicon Software Me4
#define PCO_INTERFACE_USB3    8         // USB 3.0 and USB 3.1 Gen1
#define PCO_INTERFACE_WLAN    9         // WLan (Only control path, not data path)
#define PCO_INTERFACE_CLHS   11         // Cameralink HS 
//#define PCO_INTERFACE_CLHS_SISO   12    // reserved for CLHS only SISO Me5 grabber 
//#define PCO_INTERFACE_CLHS_KAYA   13    // reserved for CLHS only KAYA Komodo grabber

#define PCO_LASTINTERFACE PCO_INTERFACE_CLHS

#define PCO_INTERFACE_CL_SER 10
#define PCO_INTERFACE_GENERIC 20

//additional defines for PCO_OpenCameraEx() strOpenStruct->wInterface
//additional information must be set in strOpenStruct->dwOpenFlags[0] and for IPV6 strOpenStruct->dwOpenFlags[1]
#define PCO_INTERFACE_BY_ID           0x8001  //camera selected with id
#define PCO_INTERFACE_BY_SERIAL       0x8002  //camera selected with serial
#define PCO_INTERFACE_BY_IPV4         0x8003  //camera selected with IPV4 address
#define PCO_INTERFACE_BY_IPV6         0x8004  //camera selected with IPV6 address

#define PCO_OPENFLAG_GENERIC_IS_CAMLINK  0x0001 // In case a generic Camerlink interface is used (serial port)
// set this flag (not necessary in automatic scanning)
#define PCO_OPENFLAG_HIDE_PROGRESS       0x0002 // Hides the progress dialog when automatic scanning runs

#ifndef MATLAB
typedef struct                         // Buffer list structure for  PCO_WaitforBuffer
{
 SHORT sBufNr;
 WORD  ZZwAlignDummy;
 DWORD dwStatusDll;
 DWORD dwStatusDrv;                    // 12
}PCO_Buflist;
#endif

typedef struct
{
  WORD          wSize;                 // Sizeof this struct
  WORD          wInterfaceType;        // 1: Firewire, 2: CamLink with Matrox, 3: Genicam wrapper dll
                                       // 4: CamLink with NI, 5: GigE, 6: USB2.0, 7: CamLink with Silicon SW Me4,
                                       // 8: USB3.0, 8: WLan
  WORD          wCameraNumber;         // Start with 0 and increment till error 'No driver' is returned
                                       // Due to port occupation it might be necessary to increment this 
                                       // value a second time to get the next camera.
                                       // (e.g. pco.edge needs two CL ports).
  WORD          wCameraNumAtInterface; // Current number of camera at the interface
  WORD          wOpenFlags[10];        // [0]: moved to dwnext to position 0xFF00
                                       // [1]: moved to dwnext to position 0xFFFF0000
                                       // [2]: Bit0: PCO_OPENFLAG_GENERIC_IS_CAMLINK
                                       //            Set this bit in case of a generic Cameralink interface
                                       //            This enables the import of the additional three camera-
                                       //            link interface functions.
                                       //      Bit1: PCO_OPENFLAG_HIDE_PROGRESS
                                       //            Set this bit to disable scanner dialog

  DWORD         dwOpenFlags[5];        // [0]-[4]: moved to strCLOpen.dummy[0]-[4]
  void*         wOpenPtr[6];
  WORD          zzwDummy[8];           // 88 - 64bit: 112
}PCO_OpenStruct;

typedef struct
{
  char  szName[16];      // string with board name
  WORD  wBatchNo;        // production batch no
  WORD  wRevision;       // use range 0 to 99
  WORD  wVariant;        // variant    // 22
  WORD  ZZwDummy[20];    //            // 62
}
PCO_SC2_Hardware_DESC;

typedef struct
{
  char  szName[16];      // string with device name
  BYTE  bMinorRev;       // use range 0 to 99
  BYTE  bMajorRev;       // use range 0 to 255
  WORD  wVariant;        // variant    // 20
  WORD  ZZwDummy[22];    //            // 64
}
PCO_SC2_Firmware_DESC;

typedef struct
{
  WORD                   BoardNum;       // number of devices
  PCO_SC2_Hardware_DESC  Board[PCO_MAXVERSIONHW];// 622
}
PCO_HW_Vers;

typedef struct
{
  WORD                   DeviceNum;       // number of devices
  PCO_SC2_Firmware_DESC  Device[PCO_MAXVERSIONFW];// 642
}
PCO_FW_Vers;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wCamType;                // Camera type
  WORD        wCamSubType;             // Camera sub type
  WORD        ZZwAlignDummy1;
  DWORD       dwSerialNumber;          // Serial number of camera // 12
  DWORD       dwHWVersion;             // Hardware version number
  DWORD       dwFWVersion;             // Firmware version number
  WORD        wInterfaceType;          // Interface type          // 22
  PCO_HW_Vers strHardwareVersion;      // Hardware versions of all boards // 644
  PCO_FW_Vers strFirmwareVersion;      // Firmware versions of all devices // 1286
  WORD        ZZwDummy[39];                                       // 1364
} PCO_CameraType;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        ZZwAlignDummy1;
  PCO_CameraType strCamType;           // previous described structure // 1368
  DWORD       dwCamHealthWarnings;     // Warnings in camera system
  DWORD       dwCamHealthErrors;       // Errors in camera system
  DWORD       dwCamHealthStatus;       // Status of camera system      // 1380
  SHORT       sCCDTemperature;         // CCD temperature in 0.1 degree Celsius
  SHORT       sCamTemperature;         // Camera temperature           // 1384
  SHORT       sPowerSupplyTemperature; // Power device temperature
  WORD        ZZwDummy[37];                                            // 1460
} PCO_General;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wSensorTypeDESC;         // Sensor type (Bit 0: true -> color sensor; false monochrome sensor)
  WORD        wSensorSubTypeDESC;      // Sensor subtype
  WORD        wMaxHorzResStdDESC;      // Maxmimum horz. resolution in std.mode
  WORD        wMaxVertResStdDESC;      // Maxmimum vert. resolution in std.mode // 10
  WORD        wMaxHorzResExtDESC;      // Maxmimum horz. resolution in ext.mode
  WORD        wMaxVertResExtDESC;      // Maxmimum vert. resolution in ext.mode
  WORD        wDynResDESC;             // Dynamic resolution of ADC in bit
  WORD        wMaxBinHorzDESC;         // Maxmimum horz. binning
  WORD        wBinHorzSteppingDESC;    // Horz. bin. stepping (0:bin, 1:lin)    // 20
  WORD        wMaxBinVertDESC;         // Maxmimum vert. binning
  WORD        wBinVertSteppingDESC;    // Vert. bin. stepping (0:bin, 1:lin)
  WORD        wRoiHorStepsDESC;        // Minimum granularity of ROI in pixels
  WORD        wRoiVertStepsDESC;       // Minimum granularity of ROI in pixels
  WORD        wNumADCsDESC;            // Number of ADCs in system              // 30
  WORD        wMinSizeHorzDESC;        // Minimum x-size in pixels in horz. direction
  DWORD       dwPixelRateDESC[4];      // Possible pixelrate in Hz              // 48
  DWORD       ZZdwDummypr[20];                                                  // 128
  WORD        wConvFactDESC[4];        // Possible conversion factor in e/cnt   // 136
  SHORT       sCoolingSetpoints[10];   // Cooling setpoints in case there is no cooling range // 156
  WORD        ZZwDummycv[8];                                                    // 172
  WORD        wSoftRoiHorStepsDESC;    // Minimum granularity of SoftROI in pixels
  WORD        wSoftRoiVertStepsDESC;   // Minimum granularity of SoftROI in pixels
  WORD        wIRDESC;                 // IR enhancment possibility
  WORD        wMinSizeVertDESC;        // Minimum y-size in pixels in vert. direction
  DWORD       dwMinDelayDESC;          // Minimum delay time in ns
  DWORD       dwMaxDelayDESC;          // Maximum delay time in ms
  DWORD       dwMinDelayStepDESC;      // Minimum stepping of delay time in ns  // 192
  DWORD       dwMinExposureDESC;       // Minimum exposure time in ns
  DWORD       dwMaxExposureDESC;       // Maximum exposure time in ms           // 200
  DWORD       dwMinExposureStepDESC;   // Minimum stepping of exposure time in ns
  DWORD       dwMinDelayIRDESC;        // Minimum delay time in ns
  DWORD       dwMaxDelayIRDESC;        // Maximum delay time in ms              // 212
  DWORD       dwMinExposureIRDESC;     // Minimum exposure time in ns
  DWORD       dwMaxExposureIRDESC;     // Maximum exposure time in ms           // 220
  WORD        wTimeTableDESC;          // Timetable for exp/del possibility
  WORD        wDoubleImageDESC;        // Double image mode possibility
  SHORT       sMinCoolSetDESC;         // Minimum value for cooling
  SHORT       sMaxCoolSetDESC;         // Maximum value for cooling
  SHORT       sDefaultCoolSetDESC;     // Default value for cooling             // 230
  WORD        wPowerDownModeDESC;      // Power down mode possibility
  WORD        wOffsetRegulationDESC;   // Offset regulation possibility
  WORD        wColorPatternDESC;       // Color pattern of color chip
                                       // four nibbles (0,1,2,3) in word 
                                       //  ----------------- 
                                       //  | 3 | 2 | 1 | 0 |
                                       //  ----------------- 
                                       //   
                                       // describe row,column  2,2 2,1 1,2 1,1
                                       // 
                                       //   column1 column2
                                       //  ----------------- 
                                       //  |       |       |
                                       //  |   0   |   1   |   row1
                                       //  |       |       |
                                       //  -----------------
                                       //  |       |       |
                                       //  |   2   |   3   |   row2
                                       //  |       |       |
                                       //  -----------------
                                       // Upper left is
                                       // Red: 0x4321; Green (red line): 0x3412; Green (blue line): 0x2143; Blue 0x1234
                                       // 0 for monochrome camera
  WORD        wPatternTypeDESC;        // Pattern type of color chip
                                       // 0: monochrome sensor
                                       // 1: RGB Bayer pattern
  WORD        wDummy1;                 // former DSNU correction mode             // 240
  WORD        wDummy2;                 // 
  WORD        wNumCoolingSetpoints;    //
  DWORD       dwGeneralCapsDESC1;      // General capabilities:
                                       // Bit 0: Noisefilter available
                                       // Bit 1: Hotpixelfilter available
                                       // Bit 2: Hotpixel works only with noisefilter
                                       // Bit 3: Timestamp ASCII only available (Timestamp mode 3 enabled)

                                       // Bit 4: Dataformat 2x12
                                       // Bit 5: Record Stop Event available
                                       // Bit 6: Hot Pixel correction
                                       // Bit 7: Ext.Exp.Ctrl. not available

                                       // Bit 8: Timestamp not available
                                       // Bit 9: Acquire mode not available
                                       // Bit10: Dataformat 4x16
                                       // Bit11: Dataformat 5x16

                                       // Bit12: Camera has no internal recorder memory
                                       // Bit13: Camera can be set to fast timing mode (PIV)
                                       // Bit14: Camera can produce metadata
                                       // Bit15: Camera allows Set/GetFrameRate cmd

                                       // Bit16: Camera has Correlated Double Image Mode
                                       // Bit17: Camera has CCM
                                       // Bit18: Camera can be synched externally
                                       // Bit19: Global shutter setting not available

                                       // Bit20: Camera supports global reset rolling readout
                                       // Bit21: Camera supports extended acquire command
                                       // Bit22: Camera supports fan control command
                                       // Bit23: Camera vert.ROI must be symmetrical to horizontal axis

                                       // Bit24: Camera horz.ROI must be symmetrical to vertical axis
                                       // Bit25: Camera has cooling setpoints instead of cooling range

                                       // Bit26: 
                                       // Bit27: reserved for future use
                                       
                                       // Bit28: reserved for future desc.// Bit29:  reserved for future desc.

                                       // Bit 30: HW_IO_SIGNAL_DESCRIPTOR available
                                       // Bit 31: Enhanced descriptor available

  DWORD       dwGeneralCapsDESC2;      // General capabilities 2                  // 252
                                       // Bit 0 ... 29: reserved for future use
                                       // Bit 30: used internally (sc2_defs_intern.h)
                                       // Bit 31: used internally (sc2_defs_intern.h)
  DWORD       dwExtSyncFrequency[4];   // lists four frequencies for external sync feature
  DWORD       dwGeneralCapsDESC3;      // general capabilites descr. 3 
  DWORD       dwGeneralCapsDESC4;      // general capabilites descr. 4            // 276
  DWORD       ZZdwDummy[40];                                                      // 436
} PCO_Description;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        ZZwAlignDummy1;
  DWORD       dwMinPeriodicalTimeDESC2;// Minimum periodical time tp in (nsec)
  DWORD       dwMaxPeriodicalTimeDESC2;// Maximum periodical time tp in (msec)        (12)
  DWORD       dwMinPeriodicalConditionDESC2;// System imanent condition in (nsec)
                                       // tp - (td + te) must be equal or longer than
                                       // dwMinPeriodicalCondition
  DWORD       dwMaxNumberOfExposuresDESC2;// Maximum number of exporures possible        (20)
  LONG        lMinMonitorSignalOffsetDESC2;// Minimum monitor signal offset tm in (nsec)
                                       // if(td + tstd) > dwMinMon.)
                                       //   tm must not be longer than dwMinMon
                                       // else
                                       //   tm must not be longer than td + tstd
  DWORD       dwMaxMonitorSignalOffsetDESC2;// Maximum -''- in (nsec)                      
  DWORD       dwMinPeriodicalStepDESC2;// Minimum step for periodical time in (nsec)  (32)
  DWORD       dwStartTimeDelayDESC2;   // Minimum monitor signal offset tstd in (nsec)
                                       // see condition at dwMinMonitorSignalOffset
  DWORD       dwMinMonitorStepDESC2;   // Minimum step for monitor time in (nsec)     (40)
  DWORD       dwMinDelayModDESC2;      // Minimum delay time for modulate mode in (nsec)
  DWORD       dwMaxDelayModDESC2;      // Maximum delay time for modulate mode in (msec)
  DWORD       dwMinDelayStepModDESC2;  // Minimum delay time step for modulate mode in (nsec)(52)
  DWORD       dwMinExposureModDESC2;   // Minimum exposure time for modulate mode in (nsec)
  DWORD       dwMaxExposureModDESC2;   // Maximum exposure time for modulate mode in (msec)(60)
  DWORD       dwMinExposureStepModDESC2;// Minimum exposure time step for modulate mode in (nsec)
  DWORD       dwModulateCapsDESC2;     // Modulate capabilities descriptor
  DWORD       dwReserved[16];                                                         //(132)
  DWORD       ZZdwDummy[41];                                                          // 296
} PCO_Description2;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wChannelNumberIntensifiedDESC; // 0: Master channel; 1x: Slave channels
  WORD        wNumberOfChannelsIntensifiedDESC; // Number of active channels in this camera

  WORD        wMinVoltageIntensifiedDESC; // Min voltage for MCP, usually ~700V (GaAs, ~600V)
  WORD        wMaxVoltageIntensifiedDESC; // Max voltage for MCP, usually ~1100V (GaAs, ~900V)
  WORD        wVoltageStepIntensifiedDESC; // Voltage step for MCP, usually 10V
  WORD        wExtendedMinVoltageIntensifiedDESC; // Extended min voltage for MCP, 600V (GaAs, ~500V)

  WORD        wMaxLoopCountIntensifiedDESC;  // Maximum loop count for multi exposure (16)
  DWORD       dwMinPhosphorDecayIntensified_ns_DESC; // Minimum decay time in (nsec)
  DWORD       dwMaxPhosphorDecayIntensified_ms_DESC; // Maximum decay time in (msec)        (24)

  DWORD       dwFlagsIntensifiedDESC;        // Flags which gating modes are supported        (28)
                                       // 0x0001: Gating mode 1 (switch off MCP after and till next exposure)
                                       // 0x0002: Gating mode 2 (switch off MCP and on when a trigger signal is detected)
  char        szIntensifierTypeDESC[24];     // Type of image intensifier;

  // dwMCP_Rectangle??_DESC describes the position of the rectangle including the MCP circle area 
  //   referenced to the sensor format which is greater. Note that the data in 1/100 pixel reso-
  //   lution, thus you have to divide the values by 100 to get the pixel coordinate
  // If data is not valid, all values are 0x80000000!

  DWORD       dwMCP_RectangleXL_DESC;        // rectangle of the MCP circle area, x left
  DWORD       dwMCP_RectangleXR_DESC;        // rectangle of the MCP circle area, x right 
  DWORD       dwMCP_RectangleYT_DESC;        // rectangle of the MCP circle area, y top
  DWORD       dwMCP_RectangleYB_DESC;        // rectangle of the MCP circle area, y bottom (68)

  DWORD       ZZdwDummy[23];                                                          //(160)
} PCO_Description_Intensified;

typedef struct
{
  WORD        wSize;

  WORD        wDarkOffsetDESC3;             // in counts (4)

  DWORD       dwGeneralCapsDESC5;           // 
  DWORD       dwGeneralCapsDESC6;           // 
  DWORD       dwGeneralCapsDESC7;           // 
  DWORD       dwGeneralCapsDESC8;           // (20)

  WORD        wMinHorzResStdDESC3;          // Minimum horz. resolution in std.mode
  WORD        wMinVertResStdDESC3;          // Minimum vert. resolution in std.mode
  WORD        wMinHorzResExtDESC3;          // Minimum horz. resolution in ext.mode
  WORD        wMinVertResExtDESC3;          // Minimum vert. resolution in ext.mode

  WORD        wPixelsize_horzDESC3;         // in nanometer
  WORD        wPixelsize_vertDESC3;         // in nanometer (32)

  SHORT       sMinSensorTempWarningDESC3;   // lower bound Value in 1/10 degree Celsius which does set WARNING_SENSORTEMPERATURE Bit in Health Status
  SHORT       sMaxSensorTempWarningDESC3;   // upper bound Value in 1/10 degree Celsius which does set WARNING_SENSORTEMPERATURE Bit in Health Status
  SHORT       sMinCameraTempWarningDESC3;   // lower bound Value in 1/10 degree Celsius which does set WARNING_CAMERATEMPERATURE Bit in Health Status
  SHORT       sMaxCameraTempWarningDESC3;   // upper bound Value in 1/10 degree Celsius which does set WARNING_CAMERATEMPERATURE Bit in Health Status
  SHORT       sMinPowerTempWarningDESC3;    // lower bound Value in 1/10 degree Celsius which does set WARNING_POWERSUPPLYTEMPERATURE Bit in Health Status
  SHORT       sMaxPowerTempWarningDESC3;    // upper bound Value in 1/10 degree Celsius which does set WARNING_POWERSUPPLYTEMPERATURE Bit in Health Status (44)

  WORD        wMinPowerVoltageWarningDESC3; // lower bound Value in 1/10 Volt which does set WARNING_POWERSUPPLYVOLTAGERANGE Bit in Health Status
  WORD        wMaxPowerVoltageWarningDESC3; // upper bound Value in 1/10 Volt which does set WARNING_POWERSUPPLYVOLTAGERANGE Bit in Health Status (48)

  SHORT       sMinSensorTempErrorDESC3;     // lower bound Value in 1/10 degree Celsius which does set ERROR_SENSORTEMPERATURE Bit in Health Status
  SHORT       sMaxSensorTempErrorDESC3;     // upper bound Value in 1/10 degree Celsius which does set ERROR_SENSORTEMPERATURE Bit in Health Status
  SHORT       sMinCameraTempErrorDESC3;     // lower bound Value in 1/10 degree Celsius which does set ERROR_CAMERATEMPERATURE Bit in Health Status
  SHORT       sMaxCameraTempErrorDESC3;     // upper bound Value in 1/10 degree Celsius which does set ERROR_CAMERATEMPERATURE Bit in Health Status
  SHORT       sMinPowerTempErrorDESC3;      // lower bound Value in 1/10 degree Celsius which does set ERROR_POWERSUPPLYTEMPERATURE Bit in Health Status
  SHORT       sMaxPowerTempErrorDESC3;      // upper bound Value in 1/10 degree Celsius which does set ERROR_POWERSUPPLYTEMPERATURE Bit in Health Status (60)

  WORD        wMinPowerVoltageErrorDESC3;   // lower bound Value in 1/10 Volt which does set ERROR_POWERSUPPLYVOLTAGERANGE Bit in Health Status
  WORD        wMaxPowerVoltageErrorDESC3;   // upper bound Value in 1/10 Volt which does set ERROR_POWERSUPPLYVOLTAGERANGE Bit in Health Status (64)

  DWORD       dwReserved[32];               // (192)
} PCO_Description3;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
} PCO_DescriptionEx;


// Hardware IO Signals definition
// see sc2_defs.h for signal options, type, polarity and filter definitions

#define NUM_MAX_SIGNALS     20         // Maximum number of signals available
#define NUM_SIGNALS 4
#define NUM_SIGNAL_NAMES 4
#define AUTO_SELECT_TIMING_SIGNAL_7 0xA000 // Used in PCO_SetHWIOSignalTiming to auto select exp. functionality
typedef struct
{
  WORD  wSize;                         // Sizeof this (for future enhancements)
  WORD  ZZwAlignDummy1;
  char  strSignalName[NUM_SIGNAL_NAMES][25];// Name of signal 104
                                       // Specifies NUM_SIGNAL_NAMES functionalities (1-4)
  WORD wSignalDefinitions;             // Flags showing signal options
                                       // 0x01: Signal can be enabled/disabled
                                       // 0x02: Signal is a status (output)
                                       // 0x10: Func. 1 has got timing settings
                                       // 0x20: Func. 2 has got timing settings
                                       // 0x40: Func. 3 has got timing settings
                                       // 0x80: Func. 4 has got timing settings
                                       // Rest: future use, set to zero!
  WORD wSignalTypes;                   // Flags showing the selectability of signal types
                                       // 0x01: TTL
                                       // 0x02: High Level TTL
                                       // 0x04: Contact Mode
                                       // 0x08: RS485 diff.
                                       // Rest: future use, set to zero!
  WORD wSignalPolarity;                // Flags showing the selectability
                                       // of signal levels/transitions
                                       // 0x01: High Level active
                                       // 0x02: Low Level active
                                       // 0x04: Rising edge active
                                       // 0x08: Falling edge active
                                       // Rest: future use, set to zero!
  WORD wSignalFilter;                  // Flags showing the selectability of filter
                                       // settings
                                       // 0x01: Filter can be switched off (t > ~65ns)
                                       // 0x02: Filter can be switched to medium (t > ~1us)
                                       // 0x04: Filter can be switched to high (t > ~100ms) 112
  DWORD dwDummy[22];                   // reserved for future use. (only in SDK) 200
}PCO_Single_Signal_Desc;

typedef struct
{
  WORD              wSize;             
  WORD              wNumOfSignals;     // Parameter to fetch the num. of descr. from the camera
  PCO_Single_Signal_Desc strSingeSignalDesc[NUM_MAX_SIGNALS];// Array of singel signal descriptors // 4004
  DWORD             dwDummy[524];      // reserved for future use.    // 6100
} PCO_Signal_Description;

#define PCO_SENSORDUMMY 7
typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        ZZwAlignDummy1;
  PCO_Description strDescription;      // previous described structure // 440
  PCO_Description2 strDescription2;    // second descriptor            // 736
  PCO_Description_Intensified strDescriptionIntensified;// Intensified camera descriptor        // 896
  PCO_Description3 strDescription3;    // third descriptor             // 1088
  DWORD       ZZdwDummy2[168];         //                              // 1760
  WORD        wSensorformat;           // Sensor format std/ext
  WORD        wRoiX0;                  // Roi upper left x
  WORD        wRoiY0;                  // Roi upper left y
  WORD        wRoiX1;                  // Roi lower right x
  WORD        wRoiY1;                  // Roi lower right y            // 1770
  WORD        wBinHorz;                // Horizontal binning
  WORD        wBinVert;                // Vertical binning
  WORD        wIntensifiedFlags;       // Additional Intensified flags for setup: 0x01 - Enable Extended Min Voltage for MCP
  DWORD       dwPixelRate;             // 32bit unsigend, Pixelrate in Hz: // 1780
                                       // depends on descriptor values
  WORD        wConvFact;               // Conversion factor:
                                       // depends on descriptor values
  WORD        wDoubleImage;            // Double image mode
  WORD        wADCOperation;           // Number of ADCs to use
  WORD        wIR;                     // IR sensitivity mode
  SHORT       sCoolSet;                // Cooling setpoint             // 1790
  WORD        wOffsetRegulation;       // Offset regulation mode       // 1792
  WORD        wNoiseFilterMode;        // Noise filter mode
  WORD        wFastReadoutMode;        // Fast readout mode for dimax
  WORD        wDSNUAdjustMode;         // DSNU Adjustment mode
  WORD        wCDIMode;                // Correlated double image mode // 1800
  WORD        wIntensifiedVoltage;     // MCP Voltage: ~700...~1100V
  WORD        wIntensifiedGatingMode;  // MCP gating mode 0: MCP always on; 1: MCP will be off after exp.; 2: MCP will be off till trigger
  DWORD       dwIntensifiedPhosphorDecay_us; // MCP phosphor decay (additional exp. time till phosphor light vanishes)
  WORD        ZZwDummy[32];                                            // 1872
  PCO_Signal_Description strSignalDesc;// Signal descriptor            // 7972
  DWORD       ZZdwDummy[PCO_SENSORDUMMY];                              // 8000
} PCO_Sensor;

typedef struct
{
  WORD  wSize;                         // Sizeof this struct
  WORD  wSignalNum;                    // Index for strSignal (0,1,2,3,)
  WORD  wEnabled;                      // Flag shows enable state of the signal (0: off, 1: on)
  WORD  wType;                         // Selected signal type (1: TTL, 2: HL TTL, 4: contact, 8: RS485, 80: TTL-A/GND-B)
  WORD  wPolarity;                     // Selected signal polarity (1: H, 2: L, 4: rising, 8: falling)
  WORD  wFilterSetting;                // Selected signal filter (1: off, 2: med, 4: high) // 12
  WORD  wSelected;                     // Select signal (0: standard signal, >1 other signal)
  WORD  ZZwReserved;
  DWORD dwParameter[4];                // Timing parameter for signal[wSelected]
  DWORD dwSignalFunctionality[4];      // Type of functionality behind the signal[wSelected] to select the
                                       // correct parameter set (e.g. 7->Parameter for 'Rolling Shutter exp.signal'
  DWORD ZZdwReserved[3];               // 60
} PCO_Signal;

typedef struct
{
  WORD   wSize;
  WORD   wDummy;
  DWORD  FrameTime_ns;                 // Frametime replaces COC_Runtime
  DWORD  FrameTime_s;   
  DWORD  ExposureTime_ns;
  DWORD  ExposureTime_s;               // 5
  DWORD  TriggerSystemDelay_ns;        // System internal min. trigger delay
  DWORD  TriggerSystemJitter_ns;       // Max. possible trigger jitter -0/+ ... ns
  DWORD  TriggerDelay_ns;              // Resulting trigger delay = system delay
  DWORD  TriggerDelay_s;               // + delay of SetDelayExposureTime ... // 9
  DWORD  ZZdwDummy[11];                // 80
} PCO_ImageTiming;


#define PCO_TIMINGDUMMY 24
typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wTimeBaseDelay;          // Timebase delay 0:ns, 1:us, 2:ms
  WORD        wTimeBaseExposure;       // Timebase expos 0:ns, 1:us, 2:ms
  WORD        wCMOSParameter;          // Line Time mode: 0: off 1: on    // 8
  DWORD       dwCMOSDelayLines;        // See next line
  DWORD       dwCMOSExposureLines;     // Delay and Exposure lines for line scanning mode // 16
  DWORD       dwDelayTable[PCO_MAXDELEXPTABLE];// Delay table             // 80
  DWORD       ZZdwDummy1[110];                                            // 520
  DWORD       dwCMOSLineTimeMin;       // Minimum line time in ns
  DWORD       dwCMOSLineTimeMax;       // Maximum line time in ms         // 528
  DWORD       dwCMOSLineTime;          // Current line time value         // 532
  WORD        wCMOSTimeBase;           // Current time base for line time
  WORD        wIntensifiedLoopCount;   // Number of loops to use for mutli exposure
  DWORD       dwExposureTable[PCO_MAXDELEXPTABLE];// Exposure table       // 600
  DWORD       ZZdwDummy2[110];                                            // 1040
  DWORD       dwCMOSFlags;             // Flags indicating the option, whether it is possible to LS-Mode with slow/fast scan, etc.
  DWORD       dwCMOSLineTimeMinStep;   // Minimum line time step in ns
  WORD        wTriggerMode;            // Trigger mode                    // 1050
                                       // 0: auto, 1: software trg, 2:extern 3: extern exp. ctrl
  WORD        wForceTrigger;           // Force trigger (Auto reset flag!)
  WORD        wCameraBusyStatus;       // Camera busy status 0: idle, 1: busy
  WORD        wPowerDownMode;          // Power down mode 0: auto, 1: user // 1056
  DWORD       dwPowerDownTime;         // Power down time 0ms...49,7d     // 1060
  WORD        wExpTrgSignal;           // Exposure trigger signal status
  WORD        wFPSExposureMode;        // Cmos-Sensor FPS exposure mode
  DWORD       dwFPSExposureTime;       // Resulting exposure time in FPS mode // 1068

  WORD        wModulationMode;         // Mode for modulation (0 = modulation off, 1 = modulation on) // 1070
  WORD        wCameraSynchMode;        // Camera synchronization mode (0 = off, 1 = master, 2 = slave)
  DWORD       dwPeriodicalTime;        // Periodical time (unit depending on timebase) for modulation // 1076
  WORD        wTimeBasePeriodical;     // timebase for periodical time for modulation  0 -> ns, 1 -> us, 2 -> ms
  WORD        ZZwDummy3;
  DWORD       dwNumberOfExposures;     // Number of exposures during modulation // 1084
  LONG        lMonitorOffset;          // Monitor offset value in ns      // 1088
  PCO_Signal  strSignal[NUM_MAX_SIGNALS];// Signal settings               // 2288
  WORD        wStatusFrameRate;        // Framerate status
  WORD        wFrameRateMode;          // Dimax: Mode for frame rate
  DWORD       dwFrameRate;             // Dimax: Framerate in mHz
  DWORD       dwFrameRateExposure;     // Dimax: Exposure time in ns      // 2300
  WORD        wTimingControlMode;      // Dimax: Timing Control Mode: 0->Exp./Del. 1->FPS
  WORD        wFastTimingMode;         // Dimax: Fast Timing Mode: 0->off 1->on
  WORD        ZZwDummy[PCO_TIMINGDUMMY];                                               // 2352
} PCO_Timing;

#define PCO_STORAGEDUMMY 38
typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        ZZwAlignDummy1;
  DWORD       dwRamSize;               // Size of camera ram in pages
  WORD        wPageSize;               // Size of one page in pixel       // 10
  WORD        ZZwAlignDummy4;
  DWORD       dwRamSegSize[PCO_RAMSEGCNT];// Size of ram segment 1-4 in pages // 28
  DWORD       ZZdwDummyrs[20];                                            // 108
  WORD        wActSeg;                 // no. (1 .. 4) of active segment  // 110
  WORD        wCompressionMode;        // wCompression mode for camera internal compression
  WORD        ZZwDummy[PCO_STORAGEDUMMY];                                 // 188
} PCO_Storage;


#define PCO_ACQ_CTRL_IS_EXTERN      0x00000001 // Acquire signal is set to external signal
#define PCO_ACQ_CTRL_IS_ACTIVE      0x00000002 // Acquire signal is currently active (enabled)
#define PCO_ACQ_CTRL_IS_NOT_ACTIVE  0x00000004 // Acquire signal is currently not active (disabled)

#define PCO_SET_ACQ_CTRL_EXTERN     0x00000001 // Sets acquire signal to external signal
#define PCO_SET_ACQ_CTRL_ACTIVE     0x00000002 // Sets acquire signal to active state (enabled)
#define PCO_SET_ACQ_CTRL_NOT_ACTIVE 0x00000004 // Sets acquire signal to inactive state (disabled)
#define PCO_SET_ACQ_CTRL_TRIGGER    0x00000008 // Sets acquire signal to active state and resets (causing an edge)
                                               // This can be used to issue a sequence trigger by software

#define PCO_RECORDINGDUMMY 22
typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wStorageMode;            // 0 = recorder, 1 = fifo
  WORD        wRecSubmode;             // 0 = sequence, 1 = ringbuffer
  WORD        wRecState;               // 0 = off, 1 = on
  WORD        wAcquMode;               // 0 = internal auto, 1 = external // 10
  WORD        wAcquEnableStatus;       // 0 = Acq disabled, 1 = enabled
  BYTE        ucDay;                   // MSB...LSB: day, month, year; 21.March 2003: 0x150307D3
  BYTE        ucMonth;                                                    // 14
  WORD        wYear;
  WORD        wHour;
  BYTE        ucMin;
  BYTE        ucSec;                   // MSB...LSB: h, min, s; 17:05:32 : 0x00110520 // 20
  WORD        wTimeStampMode;          // 0: no stamp, 1: stamp in first 14 pixel, 2: stamp+ASCII
  WORD        wRecordStopEventMode;    // 0: no stop event recording, 1: recording stops with event
  DWORD       dwRecordStopDelayImages; // Number of images which should pass by till stop event rises. // 28
  WORD        wMetaDataMode;           // Metadata mode 0: off, 1: meta data will be added to image data
  WORD        wMetaDataSize;           // Size of metadata in byte (number of pixel)
  WORD        wMetaDataVersion;        // Version info for metadata
  WORD        ZZwDummy1;
  DWORD       dwAcquModeExNumberImages;// Number of images for extended acquire mode
  DWORD       dwAcquModeExReserved[4]; // Reserved for future use
  WORD        ZZwDummy[PCO_RECORDINGDUMMY];                                               // 100
} PCO_Recording;

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        wXRes;                   // Res. h. = resulting horz.res.(sensor resolution, ROI, binning)
  WORD        wYRes;                   // Res. v. = resulting vert.res.(sensor resolution, ROI, binning)
  WORD        wBinHorz;                // Horizontal binning
  WORD        wBinVert;                // Vertical binning                // 10
  WORD        wRoiX0;                  // Roi upper left x
  WORD        wRoiY0;                  // Roi upper left y
  WORD        wRoiX1;                  // Roi lower right x
  WORD        wRoiY1;                  // Roi lower right y
  WORD        ZZwAlignDummy1;                                             // 20
  DWORD       dwValidImageCnt;         // no. of valid images in segment
  DWORD       dwMaxImageCnt;           // maximum no. of images in segment // 28
  WORD        wRoiSoftX0;              // SoftRoi upper left x
  WORD        wRoiSoftY0;              // SoftRoi upper left y
  WORD        wRoiSoftX1;              // SoftRoi lower right x
  WORD        wRoiSoftY1;              // SoftRoi lower right y
  WORD        wRoiSoftXRes;            // Res. h. = resulting horz.res.(softroi resolution, ROI, binning)
  WORD        wRoiSoftYRes;            // Res. v. = resulting vert.res.(softroi resolution, ROI, binning)
  WORD        wRoiSoftDouble;          // Soft ROI with double image
  WORD        ZZwDummy[33];                                               // 108
} PCO_Segment;

typedef struct
{
  WORD        wSize;                     // Sizeof this struct
  SHORT       sSaturation;               // Saturation from -100 to 100, 0 is default // 4
  SHORT       sVibrance;                 // Vibrance   from -100 to 100, 0 is default
  WORD        wColorTemp;                // Color Temperature from 2000 t0 20000 K
  SHORT       sTint;                     // Tint       from -100 to 100, 0 is default
  WORD        wMulNormR;                 // for  setting color ratio (when not using Color
  WORD        wMulNormG;                 //   Temp. and Tint! 1000 corresponds to 1.0 // 14
  WORD        wMulNormB;                 //   normalized: wMulNorm(R + G + B) / 3 = 1000!
  SHORT       sContrast;                 // Contrast   from -100 to 100, 0 is default
  WORD        wGamma;                    // Gamma * 0.01 from 40 to 250 => 0.40 to 2.5
  WORD        wSharpFixed;               // 0 = off, 100 = max.
  WORD        wSharpAdaptive;            // 0 = off, 100 = max. // 24
  WORD        wScaleMin;                 // 0 to 4095
  WORD        wScaleMax;                 // 0 to 4095
  WORD        wProcOptions;              // Processing Options as bit mask // 30
  WORD        ZZwDummy[93];                                                // 216
} PCO_Image_ColorSet;

typedef struct
{
  WORD        wSize;                     // Sizeof this struct
  WORD        ZZwAlignDummy1;                                              // 4
  PCO_Segment strSegment[PCO_RAMSEGCNT]; // Segment info                   // 436
  //PCO_Segment ZZstrDummySeg[6];          // Segment info dummy             // 1084
  //PCO_Segment strSegmentSoftROI[PCO_RAMSEGCNT];// Segment info SoftROI     // 1516
  PCO_Segment ZZstrDummySeg[14];         // Segment info dummy             // 1948
  PCO_Image_ColorSet strColorSet;        // Image conversion info          // 2164
  WORD        wBitAlignment;             // Bitalignment during readout. 0: MSB, 1: LSB aligned
  WORD        wHotPixelCorrectionMode;   // Correction mode for hotpixel
  WORD        ZZwDummy[38];                                                // 2244
} PCO_Image;

#define PCO_BUFFER_STATICS       0xFFFF0000  // Mask for all static flags
// Static flags:
#define PCO_BUFFER_ALLOCATED     0x80000000  // A buffer is allocated
#define PCO_BUFFER_EVENTDLL      0x40000000  // An event is allocated
#define PCO_BUFFER_ISEXTERN      0x20000000  // The buffer was allocated externally
#define PCO_BUFFER_EVAUTORES     0x10000000  // Set this flag to do an 'auto reset' of the
                                             // event, in case you call WaitForBuffer
// Dynamic flags:
#define PCO_BUFFER_EVENTSET      0x00008000  // The event of the buffer is set
#define PCO_BUFFER_RESETEV_DONE  0x00004000  // The auto reset of the event is done
// Informations about buffer status flags:
// 00000000 00000000 00000000 00000000
// |||||||| |||||||| |||||||| ||||||||
// ||||              ||
// ||||              |------------------- Buffer event Reset was done in WaitForBuffer
// ||||              -------------------- Buffer event is set to signaled
// ||||
// |||----------------------------------- Signaled Buffer event will be reset in WaitForBuffer
// ||------------------------------------ Buffer allocated externally
// |------------------------------------- Buffer event handle created inside DLL
// -------------------------------------- Buffer allocated

typedef struct
{
  WORD        wSize;                   // Sizeof this struct
  WORD        ZZwAlignDummy1;
  DWORD       dwBufferStatus;          // Buffer status
  HANDLE      hBufferEvent;            // Handle to buffer event  // 12 (16 @64bit)
  // HANDLE will be 8byte on 64bit OS and 4byte on 32bit OS. 
  DWORD       ZZdwBufferAddress;       // Buffer address, obsolete
  DWORD       dwBufferSize;            // Buffer size             // 20 (24 @64bit)
  DWORD       dwDrvBufferStatus;       // Buffer status in driver
  DWORD       dwImageSize;             // Image size              // 28 (32 @64bit)
  void        *pBufferAdress;          // buffer address          // 32 (40 @64bit)
#if !defined _WIN64
  DWORD       ZZdwDummyFill;           // additional dword        // 36 (40 @64bit)
#endif
  WORD        ZZwDummy[32];                                       // 100 (104 @64bit)
} PCO_APIBuffer;


#define TAKENFLAG_TAKEN       0x0001   // Device is taken by an application
#define TAKENFLAG_DEADHANDLE  0x0002   // The handle of this device is invalid because of a camera power down
                                       // or another device removal
#define TAKENFLAG_HANDLEVALID 0x0004   // The handle of this device is valid. Changed accoring to DEADHANDLE flag.

#define APIMANAGEMENTFLAG_SOFTROI_MASK  0xFEFE
#define APIMANAGEMENTFLAG_SOFTROI       0x0001 // Soft ROI is active
#define APIMANAGEMENTFLAG_SOFTROI_RESET 0x0100 // Reset Soft ROI to default camera ROI
#define APIMANAGEMENTFLAG_LINE_TIMING   0x0002 // Line timing is available
#define APIMANAGEMENTFLAG_SOFT_FPS      0x0004 // Enable setting fps in an iterative mode

#define APIMANAGEMENTFLAG_POWERCYCLE_LENSCTRL 0x0001 // In case a power on is detected this flag is set in Open (reserved for lens control)

typedef struct
{
  WORD          wSize;                 // Size of this struct
  WORD          wCameraNum;            // Current number of camera
  HANDLE        hCamera;               // Handle of the device                              // 8 (12 @64bit)
  WORD          wTakenFlag;            // Flags to show whether the device is taken or not. // 10 (14 @64bit)
  WORD          wAPIManagementFlags;   // Flags for internal use                            // 12 (16 @64bit)
  void          *pSC2IFFunc[20];       // Interface function pointer range 1                // 92 (176 @64bit)
  PCO_APIBuffer strPCOBuf[PCO_BUFCNT]; // Bufferlist (16 * 100 = 1600; 104 @64 = 1664)      // 1692 (1840 @64bit)
  PCO_APIBuffer ZZstrDummyBuf[26 - PCO_BUFCNT];// Bufferlist (10 * 100 = 1000; 104 = 1040)  // 2692 (2880 @64bit)
  void          *pSC2IFFunc2[25];      // Interface function pointer range 2                // 2792 (3088 @64bit)
#if !defined _WIN64
  DWORD dwZZdummyPtr32[23];
#endif
  DWORD         dwFrameTime4SoftFPS_ns;// Shortest frame time at full ROI ns part
  DWORD         dwFrameTime4SoftFPS_s; // Shortest frame time at full ROI s part
  SHORT         sBufferCnt;            // Index for buffer allocation
  WORD          wCameraNumAtInterface; // Current number of camera at the interface
  WORD          wInterface;            // Interface type (used before connecting to camera)
                                       // different from PCO_CameraType (!)
  WORD          wXRes;                 // X Resolution in Grabber (CamLink only)            // 2900 (3096 @64bit)
  WORD          wYRes;                 // Y Resolution in Buffer (CamLink only)             // 2902 (3098 @64bit)
  WORD          wPowerCycleFlag;       // This will be set to 0xFFFF in order to forward an eventuall power on state
  DWORD         dwIF_param[5];         // Interface specific parameter                      
                                       // 0 (FW:bandwidth or CL:baudrate ) 
                                       // 1 (FW:speed     or CL:clkfreq  ) 
                                       // 2 (FW:channel   or CL:ccline   ) 
                                       // 3 (FW:buffer    or CL:data     ) 
                                       // 4 (FW:iso_bytes or CL:transmit ) 
  WORD          wImageTransferMode;
  WORD          wRoiSoftX0;            // Soft ROI settings from PCO_SetROI call, when APIMANAGEMENTFLAG_SOFTROI flag is set
  WORD          wRoiSoftY0;
  WORD          wRoiSoftX1;
  WORD          wRoiSoftY1;
  WORD          wImageTransferParam[2];
  WORD          wImageTransferTxWidth;
  WORD          wImageTransferTxHeight;
  WORD          wInterfaceMajorVersion;
  WORD          wInterfaceMinorVersion;
  WORD          wInterfaceBuildVersion;
  WORD          wInterfacePatchVersion;                                                     // 2948 (3144 @64bit)
  WORD          wBytesPerPixel;         // Number of bytes per pixel (usually 2: WORD, can be 1: BYTE)
  DWORD         dwLineTime4SoftFPS;     // Sensor line time for soft fps
  DWORD         dwBuffersAdded;
  WORD          ZZwDummy[8];                                                               // 2976 (3172 @64bit)
} PCO_APIManagement;

typedef struct
{
  WORD              wSize;             // Sizeof this struct
  WORD              wStructRev;        // internal parameter, must be set to PCO_STRUCTDEF
  PCO_General       strGeneral;
  PCO_Sensor        strSensor;
  PCO_Timing        strTiming;
  PCO_Storage       strStorage;
  PCO_Recording     strRecording;
  PCO_Image         strImage;
  PCO_APIManagement strAPIManager;
  WORD              ZZwDummy[40];
} PCO_Camera;                          // 17404 (17600 @64Bit)


/////////////////////////////////////////////////////////////////////
/////// End: PCO_Camera structure definitions ///////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////// User Interface and Lens Control defines and structures //////
/////////////////////////////////////////////////////////////////////

 typedef struct
{
  WORD        wSize;
  WORD        wZZDummy1;
  WORD        wInterfaceID;          // number/ID of interface/device = 0, 1 to number of interfaces - 1
  WORD        wNumberOfInterfaces;   // total number of user interfaces/devices in the camera 
  WORD        wInterfaceType;        // type of interface (UART, UART unidirectional, USART, SPI, I2C, ...)
  BYTE        bBitsPerWord[4];       // any of 7, 8, 9, end of list with 0, remaining filled with 0 // 14
  DWORD       dwFrequency[12];       // possible baudrates/frequencies from low to high, 0 = end of list //62
  DWORD       dwInterfaceOptions;    // optional settings for interface (e.g. parity, unidirectional, ...)
  WORD        wInterfaceEnabled;     // enable/disable/autodisable/powersave/.... interface 
  DWORD       dwAllowedEquipment;    // flags for allowed accessories/equipment e.g. LensCtrlBirger (for CW) 
  WORD        wHandshakeType;        // bit mask, see below
  WORD        wTxBufferMaxSize;      // transmit buffer size, might be 0, in that case send function must wait until all bytes are sent!
  WORD        wRxBufferMaxSize;      // receive buffer size, might be 0, in that case send function must wait until all bytes are received! 
  WORD        wTxBufferFree;         // actual number of bytes which can be written to tx buffer, -1 for not applicable
  WORD        wRxBufferAvail;        // actual number of bytes which can be read from rx buffer, -1 for not applicable // 82
  WORD        wRFU[4];               // reserved for future use, camera will return 0, until they are defined. // 90
  WORD        wZZDummy2[25];         // reserved for future use. // 140
}PCO_UserInterfaceInfo;

typedef struct
{
  WORD        wSize;
  WORD        wZZDummy1;
  WORD        wInterfaceID;          // number/ID of selected interface/device = 0, 1 to number of interfaces - 1
  BYTE        bInterfaceEnable;      // 1 = enable / 0 = disable interface
  BYTE        bClearBuffers;         // 0x00 = do nothing, 0x01 clear rx buffer, 0x02 clear tx buffer, 0x03 clear rx and tx buffer
  DWORD       dwFrequency;           // selected frequency/bit rate // 12
  BYTE        bBitsPerWord;          // any of 7, 8, 9
  BYTE        bReserved;             // alignment filler, that following WORDs and DWORDs are aligned 
  WORD        wHandshakeType;        // any value of bitmask defined below, interface specific
  DWORD       dwInterfaceOptions;    // optional settings for interface (e.g. unidirectional, ...) // 20
  WORD        wRFU[4];               // reserved for future use, set to 0! // 28
  WORD        wZZDummy2[18];         // reserved for future use. // 64
}PCO_UserInterfaceSettings;


typedef struct
{
  WORD wSize;
  WORD wHardwareVersion;               // Hardware version queried by 'hv'
  WORD wBootloaderVersion;             // Bootloader version queried by 'bv'
  WORD wSerialNumber;                  // Serial number queried by 'sn'

  BYTE bLibraryIdentity[48];           // Full library identity string queried by 'lv'  //56
  DWORD dwLENSType;                    // This identifies the type of the lens control (Birger = 15872)
  DWORD dwStatusFlags;                 // LENSCONTROL_STATUS...
  DWORD dwInitCounter;                 // Counts number of inits in order to reflect lens changes
                                       // F numbers queried by 'da':
  DWORD dwFNumberMinimum;              // Min aperture as f/ * 10
  DWORD dwFNumberNumStops;             // Number of stops; Min aperture as f/ * 10
  DWORD dwFNumberMaximum;              // Max aperture as f/ * 10
                                       // Zoom range queried by 'dz':
  DWORD dwZoomRangeMin;                // Min zoom position
  DWORD dwZoomRangeMax;                // Max zoom position
  DWORD dwZoomPos;                     // Not used, set to zero
  DWORD dwLastZoomPos;                 // Last zoom position queried by 'gs'        //96
  DWORD dwApertures[50];               // Possible aperture values in f/ * 10       //296
  DWORD dwFocalLength;                 // Last focal length got from lens by 'lc';  //300

  LONG  lFocusMin;                     // Focus range minimum; Usually 0
  LONG  lFocusMax;                     // Focus range maximum; Usually 16383
  LONG  lFocusCurr;                    // Focus position 0...16383
  LONG  lFocusLastCurr;                // Last current focus position                //316
  WORD  wAperturePos;                  // Current aperture position
  WORD  wLastAperturePos;              // Last current aperture position
  DWORD dwfLastAperturePos;            // Last aperture position as f/ * 10          //324
}PCO_LensControlParameters;

#define LENSCONTROL_LENSTYPE_NONE         0
#define LENSCONTROL_TYPE_BIRGER           0x00B189E8 // Used for identification of LENS type

#define LENSCONTROL_STATUS_LA_CMD_DONE    0x00000001 // Indicates command 'la' was sent to lens
#define LENSCONTROL_STATUS_LENSPRESENT    0x00000002 // Indicates presence of a lens
#define LENSCONTROL_STATUS_NOAPERTURE     0x00000004 // No aperture settings are possible
#define LENSCONTROL_STATUS_MANUALFOCUS    0x00000008 // No focus settings are possible
#define LENSCONTROL_STATUS_WAITINGFORLENS 0x00000010 // Birger is here, but no lens

#define LENSCONTROL_IN_LENSVALUE_RELATIVE 0x00001000 // Set focus relative to current position

#define LENSCONTROL_OUT_LENSHITSTOP       0x00100000 // Focus movement hit a stop position
#define LENSCONTROL_OUT_LENSWASCHANGED    0x00200000 // Last focus or aperture movement caused a change 
#define LENSCONTROL_OUT_ZOOMHASCHANGED    0x00400000 // Focal length of lens has changed


typedef struct
{
  WORD        wSize;
  PCO_UserInterfaceInfo *pstrUserInterfaceInfo;
  PCO_UserInterfaceSettings* pstrUserInterfaceSettings;
  PCO_LensControlParameters* pstrLensControlParameters;
  HANDLE                     hCamera;          //18 (34 @64Bit)
}PCO_LensControl;


typedef struct  
{
  WORD wCommand;
  WORD wResult;
  WORD wType;  // 0: bytes, 1: words, 2: shorts, 3: dwords, 4: longs
  // wType tells about the union array which must be used to get the values
  union 
  {
    BYTE bArray[128];
    WORD wArray[64];
    SHORT sArray[64];
    DWORD dwArray[32];
    LONG lArray[32];
  };                                               //134
}PCO_Birger;

/////////////////////////////////////////////////////////////////////
/////// User Interface and Lens Control defines and structures //////
/////////////////////////////////////////////////////////////////////

#endif // SC2_STRUCTURES_H


