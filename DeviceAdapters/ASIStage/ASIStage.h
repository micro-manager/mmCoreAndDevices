/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

// Contributors:
// Jizhen Zhao based on code by Nenad Amodaj, 4/2007
// Modified by Nico Stuurman, 12/2007
// Refactored by Brandon Simpson, 3/2022

#ifndef _ASISTAGE_H_
#define _ASISTAGE_H_

#include "MMDevice.h"
#include "DeviceBase.h"

// ASI-specific error codes and messages
#define ERR_PORT_CHANGE_FORBIDDEN    10004
#define ERR_SET_POSITION_FAILED      10005
#define ERR_INVALID_STEP_SIZE        10006
#define ERR_INVALID_MODE             10008
#define ERR_UNRECOGNIZED_ANSWER      10009
#define ERR_UNSPECIFIED_ERROR        10010
#define ERR_NOT_LOCKED               10011
#define ERR_NOT_CALIBRATED           10012

#define ERR_OFFSET 10100 // offset when reporting error number from controller

#define ERR_UNRECOGNIZED_AXIS_PARAMETERS "Unrecognized Axis Parameters"
#define ERR_MISSING_PARAMETERS "Missing Parameters"
#define ERR_PARAMETER_OUTOF_RANGE "Parameter Out of Range"
#define ERR_UNDEFINED ERROR "Undefined Error"

#define ERR_INFO_COMMAND_NOT_SUPPORTED   10023 // can't receive output from INFO command because > 1023 characters
const char* const g_Msg_ERR_INFO_COMMAND_NOT_SUPPORTED = "Cannot use the INFO command due to Device Adapter limitations";

// external device names used used by the rest of the system to load a device from the .dll library
const char* const g_XYStageDeviceName = "XYStage";
const char* const g_ZStageDeviceName = "ZStage";
const char* const g_CRIFDeviceName = "CRIF";
const char* const g_CRISPDeviceName = "CRISP";
const char* const g_AZ100TurretName = "AZ100 Turret";
const char* const g_StateDeviceName = "State Device";
const char* const g_LEDDeviceName = "LED";
const char* const g_MagnifierDeviceName = "Magnifier";
const char* const g_TIRFDeviceName = "TIRF";

// corresponding device descriptions
const char* const g_XYStageDeviceDescription = "ASI XY stage driver adapter";
const char* const g_ZStageDeviceDescription = "ASI Z-stage driver adapter";
const char* const g_CRIFDeviceDescription = "ASI CRIF Autofocus adapter";
const char* const g_CRISPDeviceDescription = "ASI CRISP Autofocus adapter";
const char* const g_AZ100TurretDescription = "ASI AZ100 Turret Controller";
const char* const g_StateDeviceDescription = "ASI State Device";
const char* const g_LEDDeviceDescription = "ASI LED controller";
const char* const g_MagnifierDeviceDescription = "Magnifier";
const char* const g_TIRFDeviceDescription = "ASI TIRF Actuator";

// constant values
const char* const g_Open = "Open";
const char* const g_Closed = "Closed";

// CRIF states
const char* const g_CRIFState = "CRIF State";
const char* const g_CRIF_I = "Unlock (Laser Off)";
const char* const g_CRIF_L = "Laser On";
const char* const g_CRIF_Cal = "Calibrate";
const char* const g_CRIF_G = "Calibration Succeeded";
const char* const g_CRIF_B = "Calibration Failed";
const char* const g_CRIF_k = "Locking";
const char* const g_CRIF_K = "Lock";
const char* const g_CRIF_E = "Error";
const char* const g_CRIF_O = "Laser Off";

// CRISP states
const char* const g_CRISPState = "CRISP State";
const char* const g_CRISP_I = "Idle";
const char* const g_CRISP_R = "Ready";
const char* const g_CRISP_D = "Dim";
const char* const g_CRISP_K = "Lock";
const char* const g_CRISP_F = "In Focus";
const char* const g_CRISP_N = "Inhibit";
const char* const g_CRISP_E = "Error";
const char* const g_CRISP_G = "loG_cal";
const char* const g_CRISP_SG = "gain_Cal";
const char* const g_CRISP_Cal = "Calibrating";
const char* const g_CRISP_f = "Dither";
const char* const g_CRISP_C = "Curve";
const char* const g_CRISP_B = "Balance";
const char* const g_CRISP_RFO = "Reset Focus Offset";
const char* const g_CRISP_SSZ = "Save to Controller";
const char* const g_CRISP_Unknown = "Unknown";

const char* const g_CRISPOffsetPropertyName = "Lock Offset";
const char* const g_CRISPSumPropertyName = "Sum";
const char* const g_CRISPDitherErrorPropertyName = "Dither Error";
const char* const g_CRISPStatePropertyName = "CRISP State Character";

MM::DeviceDetectionStatus ASICheckSerialPort(MM::Device& device, MM::Core& core, std::string portToCheck, double answerTimeoutMs);

#endif // _ASISTAGE_H_
