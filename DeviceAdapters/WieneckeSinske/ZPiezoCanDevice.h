///////////////////////////////////////////////////////////////////////////////
// FILE:          ZPiezoCanDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske ZPiezo Controller Driver using CAN protocol
//             
//
// AUTHOR:        S3L GmbH, info@s3l.de, www.s3l.de,  08/20/2021
// COPYRIGHT:     S3L GmbH, Rosdorf, 2021
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
#ifndef _ZPIEZOCANDEVICE_H_
#define _ZPIEZOCANDEVICE_H_

#include "CAN29Axis.h"


class ZPiezoCANDevice : public CStageBase<ZPiezoCANDevice>
{
public:
	ZPiezoCANDevice(); 
	~ZPiezoCANDevice(); 


	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// MM::Stage API
	// ---------
	int SetPositionUm(double pos);
    int GetPositionUm(double& pos);
    int SetRelativePositionUm(double pos);
    double GetStepSize() {return stepSize_um_;}
    int SetPositionSteps(long steps) 
    {
      double pos_um_ = steps * stepSize_um_; 
      return  OnStagePositionChanged(pos_um_);
    }
    int GetPositionSteps(long& steps)
    {
	  double pos;
      int ret = GetPositionUm(pos);
      if (ret != DEVICE_OK)
         return ret;

      steps = (long)(pos / stepSize_um_);
      return DEVICE_OK;
    }

    int GetLimits(double& lower, double& upper);
    
    int Move(double /*v*/) {return DEVICE_OK;}

    bool IsContinuousFocusDrive() const {return false;}

  
    int Home();
	int Stop();
	int SetOrigin();
	int IsStageSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}

	
	// action interface                                                       
	// ----------------                                                       
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct); 


private:
	bool initialized_;
	double stepSize_um_;

	double answerTimeoutMs_;
	CAN29 can29_;
	CAN29Axis zAxis_;
};

#endif // _ZPIEZOCANDEVICE_H_
