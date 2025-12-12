//==============================================================================
//
// Title:      TLDC.h
// Purpose:    Thorlabs DC40 TLVISA Instrument Driver Header File
//
// Created on: 09.02.2024
// Copyright:  Thorlabs GmbH. All Rights Reserved.
//
//==============================================================================

#ifndef _TLDC_DRIVER_HEADER_
#define _TLDC_DRIVER_HEADER_

#include <vpptype.h>

#if defined(__cplusplus) || defined(__cplusplus__)
extern "C"
{
#endif

//==============================================================================
// Macros
#if defined _WIN32
    #if defined(_MSC_VER)
        #if defined(TLDC_EXPORT)
            #define TLDC_API __declspec(dllexport)
        #else
            #define TLDC_API __declspec(dllimport)
        #endif
    #else
        #if defined(TLDC_EXPORT)
            #define TLDC_API __attribute__ ((dllexport))
        #elif defined (TLDC_IMPORT)
            #define TLDC_API __attribute__ ((dllimport))
        #else
            #define TLDC_API
        #endif
    #endif
    #define TLDC_INTERNAL /**/
#else
    #if __GNUC__ >= 4
        #define TLDC_API __attribute__ ((visibility ("default")))
        #define TLDC_INTERNAL  __attribute__ ((visibility ("hidden")))
    #else
        #define TLDC_API
        #define TLDC_INTERNAL
    #endif
#endif


/*
A name in all uppercase is a macro name. Example: VI_ATTR_TERMCHAR
*/

/*========================================================================*//**
   buffer sizes
*//*=========================================================================*/
#define TLDC_BUFFER_SIZE            256     ///< General buffer size
#define TLDC_ERR_DESCR_BUFFER_SIZE  512     ///< Buffer size of error messages
#define TLDC_DATASET_NAME_SIZE       80     ///< Buffer size of data set name

/*========================================================================*//**
   Error codes for TLVISA functions
   Note:
   The instrument returns errors within the range -512 .. +1023.
   The driver adds the value VI_INSTR_ERROR_OFFSET (0xBFFC0900). So the
   driver returns instrument errors in the range 0xBFFC0700 .. 0xBFFC0CFF.
*//*=========================================================================*/
// Offsets
#undef VI_INSTR_WARNING_OFFSET
#undef VI_INSTR_ERROR_OFFSET

#define VI_INSTR_WARNING_OFFSET        (0x3FFC0900L)
#define VI_INSTR_ERROR_OFFSET          (_VI_ERROR + VI_INSTR_WARNING_OFFSET)    ///< 0xBFFC0900L

// Device errors, mirrored here in driver, this is basically a copy of the error list in device firmware
#define TL_DEV_ERR_UNKNOWN_CMD         (VI_INSTR_ERROR_OFFSET + 1L)     ///< "unknown command"
#define TL_DEV_ERR_PARAMETER           (VI_INSTR_ERROR_OFFSET + 2L)     ///< "parsing parameter error"
#define TL_DEV_ERR_P1_RANGE            (VI_INSTR_ERROR_OFFSET + 3L)     ///< "parameter 1 out of range"
#define TL_DEV_ERR_P2_RANGE            (VI_INSTR_ERROR_OFFSET + 4L)     ///< "parameter 2 out of range"
#define TL_DEV_ERR_P3_RANGE            (VI_INSTR_ERROR_OFFSET + 5L)     ///< "parameter 3 out of range"
#define TL_DEV_ERR_ONLY_IN_IDLE        (VI_INSTR_ERROR_OFFSET + 10L)    ///< "only allowed in idle mode"
#define TL_DEV_ERR_SERVICE             (VI_INSTR_ERROR_OFFSET + 11L)    ///< "service mode required"
#define TL_DEV_ERR_NOT_SVC             (VI_INSTR_ERROR_OFFSET + 12L)    ///< "not allowed in service mode"
#define TL_DEV_ERR_NOT_WHEN_ON         (VI_INSTR_ERROR_OFFSET + 13L)    ///< "not allowed in one of the ON modes (CW, TTL, MOD)"
#define TL_DEV_ERR_MATH_RANGE          (VI_INSTR_ERROR_OFFSET + 14L)    ///< "string conversion failed due to overrange"
#define TL_DEV_ERR_NOT_AUTH            (VI_INSTR_ERROR_OFFSET + 15L)    ///< "not authenticated, permission denied"
#define TL_DEV_ERR_AUTH_FAIL           (VI_INSTR_ERROR_OFFSET + 16L)    ///< "attempt to get authenticated failed"
#define TL_DEV_ERR_NVMEM               (VI_INSTR_ERROR_OFFSET + 17L)    ///< "NVMEM access not possible"
#define TL_DEV_ERR_LED_NOTFOUND        (VI_INSTR_ERROR_OFFSET + 19L)    ///< "LED memory read failed"
/* following errors should usually never show up at interface */
#define TL_DEV_ERR_HAL_ERR             (VI_INSTR_ERROR_OFFSET + 30L)    ///< "internal error: HAL error - please report"
#define TL_DEV_ERR_HAL_BUSY            (VI_INSTR_ERROR_OFFSET + 31L)    ///< "internal error: HAL busy - please report"
#define TL_DEV_ERR_HAL_TIMO            (VI_INSTR_ERROR_OFFSET + 32L)    ///< "internal error: HAL timeout - please report"
#define TL_DEV_ERR_USB                 (VI_INSTR_ERROR_OFFSET + 33L)    ///< "internal error: USB library - please report"
#define TL_DEV_ERR_NYI                 (VI_INSTR_ERROR_OFFSET + 98L)    ///< "implementation not finished"
#define TL_DEV_ERR_SNH                 (VI_INSTR_ERROR_OFFSET + 99L)    ///< "program error - please report"

// Driver errors, these are returned by driver itself without device communication
#define TL_ERR_PARAMETER               (VI_INSTR_ERROR_OFFSET + 100L)   ///< driver: "error parsing a parameter"
#define TL_INSTR_ERROR_NOT_SUPP_INTF   (VI_INSTR_ERROR_OFFSET + 101L)   ///< driver: "interface function isn't supported by this DC series type"
#define TL_ERR_INVAL_ATTR              (VI_INSTR_ERROR_OFFSET + 102L)   ///< driver: "error invalid attribute"
#define TL_ERR_INVAL_RESPONSE          (VI_INSTR_ERROR_OFFSET + 103L)   ///< driver: "error invalid response from device"


// Driver warnings
#undef VI_INSTR_WARN_OVERFLOW
#undef VI_INSTR_WARN_UNDERRUN
#undef VI_INSTR_WARN_NAN

#define VI_INSTR_WARN_OVERFLOW         (VI_INSTR_WARNING_OFFSET + 1L)   ///< 3FFC0901, 1073481985
#define VI_INSTR_WARN_UNDERRUN         (VI_INSTR_WARNING_OFFSET + 2L)   ///< 3FFC0902, 1073481986
#define VI_INSTR_WARN_NAN              (VI_INSTR_WARNING_OFFSET + 3L)   ///< 3FFC0903, 1073481987


/*========================================================================*//**
   Attributes act/min/max/default for get() functions
*//*=========================================================================*/
#define TLDC_ATTR_SET_VAL              (0)
#define TLDC_ATTR_MIN_VAL              (1)
#define TLDC_ATTR_MAX_VAL              (2)
#define TLDC_ATTR_DFLT_VAL             (3)


/*========================================================================*//**
   Temperature never measured value
*//*=========================================================================*/
#define TEMP_NEVER_MEASURED_VALUE      (NAN)   ///< 'temperature' that will be returned when selected channel never was measured before


/*========================================================================*//**
   System temperature unit macros used for TLDC_getTempUnit/TLDC_setTempUnit
*//*=========================================================================*/
#define TLDC_TEMP_U_KELVIN             (0)      ///< temperature unit Kelvin
#define TLDC_TEMP_U_CELSIUS            (1)      ///< temperature unit degree Celsius
#define TLDC_TEMP_U_FAHREN             (2)      ///< temperature unit degree Fahrenheit


/*========================================================================*//**
   Source 'state' macros used for TLDC_switchLedOutput, TLDC_getLedOutputState
*//*=========================================================================*/
#define TLDC_LED_OFF                   (0)      ///< LED off
#define TLDC_LED_ON                    (1)      ///< LED on

/*========================================================================*//**
   Source 'mode' macros used for TLDC_setLedMode, TLDC_getLedMode
*//*=========================================================================*/
#define TLDC_LED_MODE_CW               (0)      ///< DC LED mode 'CW' - Continuous Wave
#define TLDC_LED_MODE_TTL              (1)      ///< DC LED mode 'TTL'
#define TLDC_LED_MODE_MOD              (2)      ///< DC LED mode 'Modulation' (Analog)

/*========================================================================*//**
   Source macros used for TLDC_saveToNVMEM
*//*=========================================================================*/
#define TLDC_SAVE_TO_NVMEM_ALL         (0)      ///< save to NVMEM all
#define TLDC_SAVE_TO_NVMEM_LED_OPM     (1)      ///< save to NVMEM LED operating mode
#define TLDC_SAVE_TO_NVMEM_LED_CURR    (2)      ///< save to NVMEM LED currents Imax. and Iset_nvmem

/*========================================================================*//**
   Source 'status' macros used for TLDC_getDevStatus
*//*=========================================================================*/
#define STS_PWR_OKAY                (0x0001)          ///< +15V Power input normal
#define STS_INT_OKAY                (0x0002)          ///< Device internal voltages okay
#define STS_TMP_OKAY                (0x0004)          ///< Device temperature (case and board) normal
#define STS_LED_DTTD                (0x0008)          ///< Thorlabs LED detected
#define STS_DEV_OKAY                (0x00FF)          ///< Device normal
#define STS_LED_ACTV                (0x0100)          ///< LED active
#define STS_CW_MODE                 (0x0200)          ///< Device in CW mode
#define STS_TTL_MODE                (0x0400)          ///< Device in TTL mode
#define STS_ALG_MODE                (0x0800)          ///< Device in Analog or MOD mode


/*========================================================================*//**
   TLDC functions
*//*=========================================================================*/

/*========================================================================*//**
   open/close
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_init (ViRsrc resourceName, ViBoolean IDQuery, ViBoolean resetDevice, ViPSession instrumentHandle);
TLDC_API ViStatus _VI_FUNC TLDC_close (ViSession instrumentHandle);

/*========================================================================*//**
   Resource Manager TLVISA library functions
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_findRsrc (ViSession instrumentHandle, ViPUInt32 resourceCount);
TLDC_API ViStatus _VI_FUNC TLDC_getRsrcName (ViSession instrumentHandle, ViUInt32 index, ViChar _VI_FAR resourceName[]);
TLDC_API ViStatus _VI_FUNC TLDC_getRsrcInfo (ViSession instrumentHandle, ViUInt32 index, ViChar _VI_FAR modelName[], ViChar _VI_FAR serialNumber[], ViChar _VI_FAR manufacturer[], ViPBoolean resourceAvailable);


/*========================================================================*//**
   System Functions
*//*=========================================================================*/

TLDC_API ViStatus _VI_FUNC TLDC_setTempUnit (ViSession instrumentHandle, ViUInt16  temperatureUnit);
TLDC_API ViStatus _VI_FUNC TLDC_getTempUnit (ViSession instrumentHandle, ViPUInt16 temperatureUnit);


/*========================================================================*//**
   System LED Head Functions
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_getLedInfo (ViSession instrumentHandle, ViChar _VI_FAR ledName[], ViChar _VI_FAR ledSerialNumber[], ViPReal64 ledCurrentLimit, ViPReal64 ledForwardVoltage, ViPReal64 ledWavelength);


/*========================================================================*//**
   Measure Functions
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_measDeviceTemperature (ViSession instrumentHandle, ViPReal64 deviceTemperature);
TLDC_API ViStatus _VI_FUNC TLDC_measSupplyVoltage (ViSession instrumentHandle, ViPReal64 supplyVoltage);
TLDC_API ViStatus _VI_FUNC TLDC_measureLedCurrent (ViSession instrumentHandle, ViPReal64 ledCurrent);
TLDC_API ViStatus _VI_FUNC TLDC_measureLedVoltage (ViSession instrumentHandle, ViPReal64 ledVoltage);
TLDC_API ViStatus _VI_FUNC TLDC_measurePotiValue (ViSession instrumentHandle, ViPReal64 potiValue);

/*========================================================================*//**
   Source Functions
*//*=========================================================================*/

TLDC_API ViStatus _VI_FUNC TLDC_switchLedOutput (ViSession instrumentHandle, ViBoolean ledOutput);
TLDC_API ViStatus _VI_FUNC TLDC_getLedOutputState (ViSession instrumentHandle, ViPBoolean ledOutput);

/*========================================================================*//**
   Source configuration functions
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_setLedCurrentLimitUser (ViSession instrumentHandle, ViReal64 LEDCurrentLimitUser);
TLDC_API ViStatus _VI_FUNC TLDC_getLedCurrentLimitUser (ViSession instrumentHandle, ViInt16 Attribute, ViPReal64 LEDCurrentLimitUser);
TLDC_API ViStatus _VI_FUNC TLDC_setLedMode (ViSession instrumentHandle, ViUInt32  LEDMode);
TLDC_API ViStatus _VI_FUNC TLDC_getLedMode (ViSession instrumentHandle, ViPUInt32  pLEDMode);
TLDC_API ViStatus _VI_FUNC TLDC_setLedCurrentSetpoint (ViSession instrumentHandle, ViReal64 LEDCurrentSetpoint);
TLDC_API ViStatus _VI_FUNC TLDC_getLedCurrentSetpoint (ViSession instrumentHandle, ViInt16 Attribute, ViPReal64 LEDCurrentSetpoint);
TLDC_API ViStatus _VI_FUNC TLDC_setLedUseNonThorlabsLed (ViSession instrumentHandle, ViBoolean useNonThorlabsLed);
TLDC_API ViStatus _VI_FUNC TLDC_getLedUseNonThorlabsLed (ViSession instrumentHandle, ViPBoolean useNonThorlabsLed);
TLDC_API ViStatus _VI_FUNC TLDC_saveToNVMEM (ViSession instrumentHandle, ViUInt16 group);


/*========================================================================*//**
   Utility Functions
*//*=========================================================================*/
TLDC_API ViStatus _VI_FUNC TLDC_errorMessage (ViSession instrumentHandle, ViStatus errorCode, ViChar _VI_FAR errorMessage[]);
TLDC_API ViStatus _VI_FUNC TLDC_errorQuery (ViSession instrumentHandle, ViPStatus errorCode, ViChar _VI_FAR errorMessage[]);
TLDC_API ViStatus _VI_FUNC TLDC_reset (ViSession instrumentHandle);
TLDC_API ViStatus _VI_FUNC TLDC_selfTest (ViSession instrumentHandle, ViPInt16 selfTestResult, ViChar _VI_FAR selfTestMessage[]);
TLDC_API ViStatus _VI_FUNC TLDC_revisionQuery (ViSession instrumentHandle, ViChar _VI_FAR instrumentDriverRevision[], ViChar _VI_FAR firmwareRevision[]);
TLDC_API ViStatus _VI_FUNC TLDC_identificationQuery (ViSession instrumentHandle, ViChar _VI_FAR manufacturerName[], ViChar _VI_FAR deviceName[], ViChar _VI_FAR serialNumber[], ViChar _VI_FAR firmwareRevision[]);
TLDC_API ViStatus _VI_FUNC TLDC_getBuildDateAndTime (ViSession instr, ViChar _VI_FAR buildDateAndTime[]);
TLDC_API ViStatus _VI_FUNC TLDC_getCalibrationMsg (ViSession instrumentHandle, ViChar _VI_FAR message[]);
TLDC_API ViStatus _VI_FUNC TLDC_getDevStatus (ViSession instrumentHandle, ViPUInt16  devStatus);

#if defined(__cplusplus) || defined(__cplusplus__)
}
#endif

#endif  /* ndef _TLDC_DRIVER_HEADER_ */

/*- The End -----------------------------------------------------------------*/

