/*
File:		MCL_MicroDrive.h
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/
#pragma once 

#define M1AXIS 1
#define M2AXIS 2
#define M3AXIS 3
#define M4AXIS 4
#define M5AXIS 5
#define M6AXIS 6

#define BITMASK_M1 0x01
#define BITMASK_M2 0x02
#define BITMASK_M3 0x04
#define BITMASK_M4 0x08
#define BITMASK_M5 0x10
#define BITMASK_M6 0x20

#define MICRODRIVE                  0x2500
#define MICRODRIVE1					0x2501
#define MICRODRIVE3                 0x2503
#define MICRODRIVE4					0x2504
#define MICRODRIVE6					0x2506
#define NC_MICRODRIVE				0x3500
#define MADTWEEZER					0x2522

#define STANDARD_MOVE_TYPE			1
#define CALIBRATE_TYPE				2
#define HOME_TYPE					3
#define RETURN_TO_ORIGIN_TYPE		4
#define FIND_EPI_TYPE				5

#define MIN_VEL_HIGH_SPEED			1.5708
#define MAX_VEL_HIGH_SPEED			125.6637
#define MIN_VEL_HIGH_PRECISION		0.1964
#define MAX_VEL_HIGH_PRECISION		6.2831


static const char* g_StageDeviceName =	 "MicroDrive Z Stage";
static const char* g_XYStageDeviceName = "MicroDrive XY Stage";

static const char* g_Keyword_SetPosXmm = "Set position X axis (mm)";
static const char* g_Keyword_SetPosYmm = "Set position Y axis (mm)";
static const char* g_Keyword_SetPosZmm = "Set position Z axis (mm)";
static const char* g_Keyword_SetRelativePosXmm = "Move X axis (mm)";
static const char* g_Keyword_SetRelativePosYmm = "Move Y axis (mm)";
static const char* g_Keyword_SetRelativePosZmm = "Move Z axis (mm)";
static const char* g_Keyword_MaxVelocity = "Maximum velocity (mm/s)";
static const char* g_Keyword_MinVelocity = "Minimum velocity (mm/s)";
static const char* g_Keyword_SetOriginHere =  "Set origin here";
static const char* g_Keyword_Calibrate =      "Calibrate";
static const char* g_Keyword_ReturnToOrigin = "Return to origin";
static const char* g_Keyword_PositionTypeAbsRel = "Position type (absolute/relative)";
static const char* g_Keyword_Encoded = "EncodersPresent";
static const char* g_Keyword_IterativeMove = "Enable iterative moves";
static const char* g_Keyword_ImRetry = "IM number of retries";
static const char* g_Keyword_ImTolerance = "IM tolerance in Um";
static const char* g_Keyword_IsTirfModuleAxis = "TIRF module axis";
static const char* g_Keyword_IsTirfModuleAxis1 = "TIRF module axis1";
static const char* g_Keyword_IsTirfModuleAxis2 = "TIRF module axis2";	
static const char* g_Keyword_DistanceToEpi = "Distance to epi";
static const char* g_Keyword_FindEpi = "Find Epi";


// Mad Tweezer
static const char* g_DeviceMadTweezerName = "Mad-Tweezer";
static const char* ZMadTweezerName = "Mad-Tweezer Z Stage";
static const char* g_Keyword_HighSpeedMode = "High Speed";
static const char* g_Keyword_HighPrecisionMode = "High Precision";
static const char* g_Keyword_MaxVelocityHighSpeed = "Maximum high speed velocity (rad/s)";
static const char* g_Keyword_MinVelocityHighSpeed = "Minimum high speed velocity (rad/s)";
static const char* g_Keyword_MaxVelocityHighPrecision = "Maximum high precision velocity (rad/s)";
static const char* g_Keyword_MinVelocityHighPrecision = "Minimum high precision velocity (rad/s)";
static const char* g_Keyword_Home = "Home";
static const char* g_Keyword_Mode = "Mode";
static const char* g_Keyword_Direction = "Direction";
static const char* g_Keyword_Location =  "Location (milliradians)";
static const char* g_Keyword_Velocity =  "Velocity (radians/s)";
static const char* g_Keyword_WaitTime =  "Wait Time";
static const char* g_Keyword_Rotations = "Rotations";
static const char* g_Keyword_Steps =	 "Steps";
static const char* g_Keyword_Milliradians = "Milliradians";

// Common
static const char* g_Keyword_Handle = "Handle";
static const char* g_Keyword_Serial_Num = "Serial Number";
static const char* g_Keyword_ProductID = "Product ID";
static const char* g_Keyword_Stop = "Stop";

static const char* g_Listword_No = "No";
static const char* g_Listword_Yes = "Yes";
static const char* g_Listword_AbsPos = "Absolute Position";
static const char* g_Listword_RelPos = "Relative Position";
static const char* g_Listword_Clockwise = "Clockwise";
static const char* g_Listword_CounterClockwise = "Counterclockwise";


