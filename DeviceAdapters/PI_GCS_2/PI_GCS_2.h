///////////////////////////////////////////////////////////////////////////////
// FILE:          PI_GCS_2.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PI GCS Controller Driver
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/28/2006
//                Steffen Rau, s.rau@pi.ws, 28/03/2008
// COPYRIGHT:     University of California, San Francisco, 2006
//                Physik Instrumente (PI) GmbH & Co. KG, 2008
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
// CVS:           $Id: PI_GCS_2.h,v 1.9, 2014-03-31 12:51:24Z, Steffen Rau$
//

#ifndef PI_GCS_DLL_H_INCLUDED
#define PI_GCS_DLL_H_INCLUDED


#include <string>
#include <vector>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN    10004
#define ERR_UNRECOGNIZED_ANSWER      10009
#define ERR_OFFSET 10100

#ifndef WIN32
#define WINAPI
#define BOOL int
#define TRUE 1
#define FALSE 0
#endif

#define PI_MOTION_ERROR -1024L
#define COM_TIMEOUT_ERROR -7L
#define COM_ERROR -1L
#define PI_CNTR_NO_ERROR  0L
#define PI_CNTR_UNKNOWN_COMMAND 2L
#define PI_CNTR_MOVE_WITHOUT_REF_OR_NO_SERVO 5L
#define PI_CNTR_POS_OUT_OF_LIMITS  7L
#define PI_CNTR_VEL_OUT_OF_LIMITS 8L
#define PI_CNTR_INVALID_AXIS_IDENTIFIER 15L
#define PI_CNTR_PARAM_OUT_OF_RANGE 17L
#define PI_CNTR_ILLEGAL_AXIS 23L
#define PI_CNTR_AXIS_UNDER_JOYSTICK_CONTROL  51L
#define PI_CNTR_JOYSTICK_IS_ACTIVE 83L
#define PI_CNTR_MOTOR_IS_OFF 84L
#define PI_CNTR_ON_LIMIT_SWITCH 216L
#define PI_CNTR_MOTION_ERROR 1024L
#define PI_ERROR_CMD_CMD_UNKNOWN_COMMAND 49155L
#define PI_ERROR_PARAM_CMD_VALUE_OUT_OF_RANGE 49166L
#define PI_ERROR_MOT_CMD_AXIS_DISABLED 49173L
#define PI_ERROR_MOT_MOT_AXIS_NOT_REF 81938L
#define PI_ERROR_MOT_CMD_TARGET_OUT_OF_RANGE 49172L
#define PI_ERROR_MOT_CMD_INVALID_MODE_OF_OPERATION 49169L
#define PI_ERROR_MOT_CMD_AXIS_IN_FAULT 49224L

#define ERR_GCS_PI_CNTR_POS_OUT_OF_LIMITS 102
#define ERR_GCS_PI_CNTR_MOVE_WITHOUT_REF_OR_NO_SERVO 103
#define ERR_GCS_PI_CNTR_AXIS_UNDER_JOYSTICK_CONTROL 104
#define ERR_GCS_PI_CNTR_INVALID_AXIS_IDENTIFIER 105
#define ERR_GCS_PI_CNTR_ILLEGAL_AXIS 106
#define ERR_GCS_PI_CNTR_VEL_OUT_OF_LIMITS 107
#define ERR_GCS_PI_CNTR_ON_LIMIT_SWITCH 108
#define ERR_GCS_PI_CNTR_MOTION_ERROR 109
#define ERR_GCS_PI_MOTION_ERROR 110
#define ERR_GCS_PI_CNTR_PARAM_OUT_OF_RANGE 111
#define ERR_GCS_PI_NO_CONTROLLER_FOUND 112
#define ERR_DLL_PI_DLL_NOT_FOUND 113
#define ERR_DLL_PI_INVALID_INTERFACE_NAME 114
#define ERR_DLL_PI_INVALID_INTERFACE_PARAMETER 115
#define ERR_GCS_PI_AXIS_DISABLED 116
#define ERR_GCS_PI_INVALID_MODE_OF_OPERATION 117
#define ERR_GCS_PI_PARAM_VALUE_OUT_OF_RANGE 118
#define ERR_GCS_PI_CNTR_MOTOR_IS_OFF 119
#define ERR_GCS_PI_AXIS_IN_FAULT 120

extern const char* g_msg_CNTR_POS_OUT_OF_LIMITS;
extern const char* g_msg_CNTR_MOVE_WITHOUT_REF_OR_NO_SERVO;
extern const char* g_msg_CNTR_AXIS_UNDER_JOYSTICK_CONTROL;
extern const char* g_msg_CNTR_INVALID_AXIS_IDENTIFIER;
extern const char* g_msg_CNTR_ILLEGAL_AXIS;
extern const char* g_msg_CNTR_VEL_OUT_OF_LIMITS;
extern const char* g_msg_CNTR_ON_LIMIT_SWITCH;
extern const char* g_msg_CNTR_MOTION_ERROR;
extern const char* g_msg_MOTION_ERROR;
extern const char* g_msg_CNTR_PARAM_OUT_OF_RANGE;
extern const char* g_msg_NO_CONTROLLER_FOUND;
extern const char* g_msg_DLL_NOT_FOUND;
extern const char* g_msg_INVALID_INTERFACE_NAME;
extern const char* g_msg_INVALID_INTERFACE_PARAMETER;
extern const char* g_msg_AXIS_DISABLED;
extern const char* g_msg_INVALID_MODE_OF_OPERATION;
extern const char* g_msg_PARAM_VALUE_OUT_OF_RANGE;
extern const char* g_msg_CNTR_MOTOR_IS_OFF;
extern const char* g_msg_AXIS_IN_FAULT;

bool GetValue (const std::string& sMessage, double& dval);
bool GetValue (const std::string& sMessage, long& lval);
bool GetValue (const std::string& sMessage, unsigned long& lval);
std::string ExtractValue (const std::string& sMessage);
std::vector<std::string> Tokenize (const std::string& lines);
int TranslateError (int err);

#endif //PI_GCS_DLL_H_INCLUDED
