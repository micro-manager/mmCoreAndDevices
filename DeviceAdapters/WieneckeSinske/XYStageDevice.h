///////////////////////////////////////////////////////////////////////////////
// FILE:          XYStageDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske Stage Controller Driver
//                XY Stage
//             
//
// AUTHOR:        S3L GmbH, info@s3l.de, www.s3l.de,  11/21/2017
// COPYRIGHT:     S3L GmbH, Rosdorf, 2017
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation.
//                
//                You should have received a copy of the GNU Lesser General Public
//                License along with the source distribution; if not, write to
//                the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
//                Boston, MA  02111-1307  USA
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.  
//
#ifndef _XYSTAGEDEVICE_H_
#define _XYSTAGEDEVICE_H_

#include "CAN29Axis.h"


class XYStageDevice : public CXYStageBase<XYStageDevice>
{
public:
	XYStageDevice(); 
	~XYStageDevice(); 

	// Device API
	// ---------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax); 
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax); 
	int SetPositionSteps(long xSteps, long ySteps);
	int SetRelativePositionSteps(long xSteps, long ySteps);

	int GetPositionSteps(long& xSteps, long& ySteps);
	int Home();
	int Stop();
	int SetOrigin();
	double GetStepSizeXUm() {return stepSize_um_;}
	double GetStepSizeYUm() {return stepSize_um_;}
	int IsXYStageSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}

	
	// action interface                                                       
	// ----------------                                                       
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct); 
	int OnTrajectoryVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTrajectoryAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
	bool initialized_;
	double stepSize_um_;

	double answerTimeoutMs_;
	CAN29 can29_;
	CAN29Axis xAxis_;
	CAN29Axis yAxis_;
    CAN29Byte velocity_;
 
};

#endif // _XYSTAGEDEVICE_H_
