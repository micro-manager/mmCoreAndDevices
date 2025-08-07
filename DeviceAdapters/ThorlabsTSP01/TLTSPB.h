/****************************************************************************

   Thorlabs TSPB Series VISA instrument driver

   This driver supports TSP environment measurement devices

   FOR DETAILED DESCRIPTION OF THE DRIVER FUNCTIONS SEE THE ONLINE HELP FILE
   AND THE PROGRAMMERS REFERENCE MANUAL.

   Copyright:  Copyright(c) 2008-2017, Thorlabs (www.thorlabs.com)

   Disclaimer:

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
****************************************************************************/


#ifndef _TLTSPB_DRIVER_HEADER_
#define _TLTSPB_DRIVER_HEADER_

#include <stdint.h>
#include "vpptype.h"
#include "TLTSP_Defines.h"
#if defined(__cplusplus) || defined(__cplusplus__)
extern "C"
{
#endif


/*===========================================================================

 Macros

===========================================================================*/

#define TLTSP_TEMPER_CHANNEL_1       		 (11)
#define TLTSP_TEMPER_CHANNEL_2   	 		 (12)
#define TLTSP_TEMPER_CHANNEL_3				 (13)


#define VI_DRIVER_ERROR_OFFSET          (_VI_ERROR + VI_INSTR_WARNING_OFFSET - 1024)   //0xBFFBF8DC


//User defined driver error codes
#define VI_DRIVER_USB_ERROR			   (VI_DRIVER_ERROR_OFFSET)
#define VI_DRIVER_FRAME_SERV_ERROR     (VI_DRIVER_ERROR_OFFSET - 1)
#define VI_DRIVER_DEV_INTER_BROKEN     (VI_DRIVER_ERROR_OFFSET - 2)

//User defined device error codes
#define VI_INSTR_ERR_IO				   (VI_INSTR_ERROR_OFFSET)
#define VI_INSTR_ERR_RUNTIME		   (VI_INSTR_ERROR_OFFSET - 1)
#define VI_INSTR_ERR_NO_SERVICE_MODE   (VI_INSTR_ERROR_OFFSET - 2)
#define VI_INSTR_ERR_AUTH_FAILED	   (VI_INSTR_ERROR_OFFSET - 3)
#define VI_INSTR_ERR_HW_ERROR		   (VI_INSTR_ERROR_OFFSET - 4)
#define VI_INSTR_ERR_NO_SUCH_SENSOR	   (VI_INSTR_ERROR_OFFSET - 5)
#define VI_INSTR_ERR_PARAM			   (VI_INSTR_ERROR_OFFSET - 6)

#if defined(_WIN64)
	#define CALL_CONV            __fastcall
#elif (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(_NI_mswin16_)
	#define CALL_CONV            __stdcall
#endif

#ifdef _CVI_
	#undef _VI_FUNC
	#define _VI_FUNC
#elif defined IS_DLL_TARGET
	#undef _VI_FUNC
	#define _VI_FUNC __declspec(dllexport) CALL_CONV
#elif defined IS_LINUX_LIB_TARGET
	#undef _VI_FUNC
	#define _VI_FUNC
#elif defined BUILDING_DEBUG_EXE
	#undef _VI_FUNC
	#define _VI_FUNC
#else
	#undef _VI_FUNC
	#define _VI_FUNC __declspec(dllimport) CALL_CONV
#endif

#ifndef isnan
#define isnan IsNotANumber
#endif

#ifndef isinf
#define isinf  	IsInfinity
#endif


void initTLTSPB_System(void* module);

/// ------------------------------  I N I T I A L I Z A T O N ---------------------------------

ViStatus _VI_FUNC TLTSPB_findRsrc (ViSession vi, ViPUInt32 deviceCount);

ViStatus _VI_FUNC TLTSPB_getRsrcName  (ViSession vi,
                                                 ViUInt32 deviceIndex,
                                                 ViChar resourceName[]);


ViStatus _VI_FUNC TLTSPB_init (ViRsrc resourceName, ViBoolean IDQuery,
                              ViBoolean resetDevice, ViPSession vi);

ViStatus _VI_FUNC TLTSPB_getRsrcInfo (ViSession vi, ViUInt32 index, ViChar modelName[],
						ViChar serialNumber[], ViChar manufacturer[], ViPBoolean resourceInUse);

ViStatus _VI_FUNC TLTSPB_close (ViSession vi);

/// ------------------------------  C O N F I G U R A T I O N ---------------------------------

ViStatus _VI_FUNC TLTSPB_setTempSensOffset (ViSession vi, ViUInt16 channel, ViReal64 temperatureOffset);
ViStatus _VI_FUNC TLTSPB_getTempSensOffset (ViSession vi, ViUInt16 channel, ViInt16 attribute, ViPReal64 temperatureOffset);

ViStatus _VI_FUNC TLTSPB_setThermistorExpParams (ViSession vi,
                                                ViUInt16 channel,
                                                ViReal64 r0Value,
                                                ViReal64 t0Value,
                                                ViReal64 betaValue);

ViStatus _VI_FUNC TLTSPB_getThermistorExpParams (ViSession vi,
                                                ViUInt16 channel,
                                                ViInt16 attribute,
                                                ViPReal64 r0Value,
                                                ViPReal64 t0Value,
                                                ViPReal64 betaValue);

ViStatus _VI_FUNC TLTSPB_setHumSensOffset (ViSession vi, ViReal64 humidityOffset);
ViStatus _VI_FUNC TLTSPB_getHumSensOffset (ViSession vi, ViInt16 attribute, ViPReal64 humidityOffset);

/// ------------------------  M E A S U R E   F U N C T I O N S -------------------------------

ViStatus _VI_FUNC TLTSPB_getTemperatureData (ViSession vi,ViUInt16 channel, ViInt16 attribute, ViPReal64 temperatureValue);

ViStatus _VI_FUNC TLTSPB_getHumidityData (ViSession vi, ViInt16 attribute, ViPReal64 humidityValue);

ViStatus _VI_FUNC TLTSPB_getThermRes (ViSession vi, ViUInt16 channel, ViInt16 attribute, ViPReal64 resistanceInOhm);

ViStatus _VI_FUNC TLTSPB_getConfiguration (ViSession vi, ViPInt16 configuration);

ViStatus _VI_FUNC TLTSPB_measTemperature (ViSession vi, ViUInt16 channel, ViPReal64 temperature);

ViStatus _VI_FUNC TLTSPB_measHumidity (ViSession vi, ViPReal64 humidity);

/// ------------------------  U T I L I T Y   F U N C T I O N S -------------------------------

ViStatus _VI_FUNC TLTSPB_identificationQuery (ViSession vi,
                                             ViChar manufacturerName[],
                                             ViChar deviceName[],
                                             ViChar serialNumber[],
                                             ViChar firmwareRevision[]);

ViStatus _VI_FUNC TLTSPB_reset (ViSession vi);

ViStatus _VI_FUNC TLTSPB_selfTest (ViSession vi, ViPInt16 selfTestResult, ViChar description[]);

ViStatus _VI_FUNC TLTSPB_errorQuery (ViSession vi, ViPInt32 errorNumber, ViChar errorMessage[]);

ViStatus _VI_FUNC TLTSPB_errorMessage (ViSession vi, ViStatus statusCode, ViChar description[]);

ViStatus _VI_FUNC TLTSPB_getProductionDate (ViSession vi,
                                           ViChar productionDate[]);
ViStatus _VI_FUNC TLTSPB_revisionQuery (ViSession vi, ViChar instrumentDriverRevision[], ViChar firmwareRevision[]);

/// ------------------------------  S E R V I C E   M O D E ---------------------------------

ViStatus _VI_FUNC TLTSPB_setServiceMode (ViSession vi, ViChar password[]);

ViStatus _VI_FUNC TLTSPB_isServiceMode (ViSession vi, ViBoolean *serviceModeActive);

ViStatus _VI_FUNC TLTSPB_setProductionDate (ViSession vi, ViChar productionDate[]);

ViStatus _VI_FUNC TLTSPB_setDeviceName (ViSession vi, ViChar deviceName[]);

ViStatus _VI_FUNC TLTSPB_setSerialNr (ViSession vi, ViChar serialNr[]);

ViStatus _VI_FUNC TLTSPB_getUUID (ViSession instrumentHandle, ViChar uuid[]);

ViStatus _VI_FUNC TLTSPB_setExtRefVoltage (ViSession vi, double referenceVoltage);

ViStatus _VI_FUNC TLTSPB_getExtRefVoltage (ViSession vi, double *referenceVoltage);

ViStatus _VI_FUNC TLTSPB_setExtSerResist (ViSession vi, double serialResistance);

ViStatus _VI_FUNC TLTSPB_getExtSerResist (ViSession vi, double *serialResistance);

#if defined(__cplusplus) || defined(__cplusplus__)
}
#endif

#endif   /* _TLTSPB_DRIVER_HEADER_ */

