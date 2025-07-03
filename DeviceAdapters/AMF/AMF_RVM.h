///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_RVM.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Advanced MicroFluidics (AMF) Rotary
//				  Valve Modules (RVM).
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2024
//                
// VERSION:       0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    26.02.2024 LK

#ifndef _AMF_RVM_H_
#define _AMF_RVM_H_

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <map>
#include <string>
#include "AMF_Commands.h"

#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107


enum RotationDirection {
	SHORTEST,
	CLOCKWISE,
	COUNTERCLOCKWISE
};

//////////////////////////////////////////////////////////////////////////////
// AMF_RVM class
// Device adapter for AMF Rotary Valve Modules (RVM)
//////////////////////////////////////////////////////////////////////////////

class AMF_RVM : public CStateDeviceBase<AMF_RVM>
{
public:
	AMF_RVM();
	~AMF_RVM();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	bool Busy();
	unsigned long GetNumberOfPositions()const { return nPos_; };

	// Action Handlers
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChangeAddress(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRotationDirection(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Utility Members
	int GetValvePosition(int& pos);
	int SetValvePosition(int pos);
	std::string RotationDirectionToString(RotationDirection rd);
	RotationDirection RotationDirectionFromString(std::string& s);

private:
	bool initialized_ = false;
	bool busy_ = false;
	unsigned int address_ = 1; // Default address
	long nPos_ = 0;
	long position_ = 0;
	RotationDirection rotationDirection_ = SHORTEST;
	MM::MMTime changedTime_;
	std::string version_;

	// Serial utility members
	std::string port_;
	int SendRecv(AMFCommand& cmd);
};

#endif //_AMF_RVM_H_