///////////////////////////////////////////////////////////////////////////////
// FILE:          XYStageDevice.cpp
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

#ifdef WIN32
#include <windows.h>
#else
#include <arpa/inet.h>
#endif
#include "FixSnprintf.h"

#include "XYStageDevice.h"
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "WieneckeSinske.h"

#include <sstream>


///////////////////////////////////////////////////////////////////////////////

using namespace std;




///////////////////////////////////////////////////////////////////////////////
// XYStageDeviceDevice
//
XYStageDevice::XYStageDevice (): 
CXYStageBase<XYStageDevice>(),
	initialized_ (false),
	stepSize_um_(0.001),
	can29_(),
	xAxis_(CAN_XAXIS, 0, &can29_),
	yAxis_(CAN_YAXIS, 0, &can29_),
	velocity_(0)
{ 

	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_XYStageDeviceDeviceName, MM::String, true);

	// Description                                                            
	CreateProperty(MM::g_Keyword_Description, "Controller for piezo stages", MM::String, true);

	// Port                                                                   
	CPropertyAction* pAct = new CPropertyAction (this, &XYStageDevice::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

}

XYStageDevice::~XYStageDevice() 
{
	xAxis_.UnInitialize();
	yAxis_.UnInitialize();

	Shutdown();
}



bool XYStageDevice::Busy()
{	
	return xAxis_.IsBusy() || yAxis_.IsBusy();
}

void XYStageDevice::GetName (char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_XYStageDeviceDeviceName);
}


int XYStageDevice::Initialize()
{
	if (!can29_.portInitialized_)
		return ERR_DEVICE_NOT_ACTIVE;

	can29_.Initialize(this, GetCoreCallback());

	xAxis_.Initialize();
	yAxis_.Initialize();


	// check if this Axis exists:
	bool presentX, presentY;
	// TODO: check both stages
	int ret = yAxis_.GetPresent(presentY, "WSB PiezoDrive CAN");
	if (ret != DEVICE_OK)
		return ret;
	ret =  yAxis_.GetPresent(presentX, "WSB PiezoDrive CAN");
	if (ret != DEVICE_OK)
		return ret;
	if (!(presentX && presentY))
		return ERR_MODULE_NOT_FOUND;


	// set property list
	// ----------------
	// Trajectory Velocity and Acceleration:
	CPropertyAction* pAct = new CPropertyAction(this, &XYStageDevice::OnTrajectoryVelocity);
	ret = CreateProperty("Velocity (micron/s)", "0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	ret = SetPropertyLimits("Velocity (micron/s)", 0, 100000);
	if (ret != DEVICE_OK)
		return ret;
	

	pAct = new CPropertyAction(this, &XYStageDevice::OnTrajectoryAcceleration);
	ret = CreateProperty("Acceleration (micron/s^2)", "0", MM::Float, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	ret = SetPropertyLimits("Acceleration (micron/s^2)", 0, 500000);
	if (ret != DEVICE_OK)
		return ret;
	
	initialized_ = true;

	return DEVICE_OK;
}

int XYStageDevice::Shutdown()
{
	if (initialized_) initialized_ = false;
	return DEVICE_OK;
}


int XYStageDevice::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) 
{
	long xMi, xMa, yMi, yMa;
	GetStepLimits(xMi, xMa, yMi, yMa);
	xMin = xMi * stepSize_um_;
	yMin = yMi * stepSize_um_;
	xMax = xMa * stepSize_um_;
	yMax = yMa * stepSize_um_;

	return DEVICE_OK;
}

int XYStageDevice::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) 
{	
	xAxis_.GetLowerHardwareStop((CAN29Long&)xMin);
	xAxis_.GetUpperHardwareStop((CAN29Long&)xMax);
	yAxis_.GetLowerHardwareStop((CAN29Long&)yMin);
	yAxis_.GetUpperHardwareStop((CAN29Long&)yMax);

	return DEVICE_OK;
}


int XYStageDevice::SetPositionSteps(long xSteps, long ySteps)
{	
	int ret = xAxis_.SetPosition((CAN29Long)xSteps, velocity_);
	if (ret != DEVICE_OK)
		return ret;

	ret = yAxis_.SetPosition((CAN29Long)ySteps, velocity_);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int XYStageDevice::SetRelativePositionSteps(long xSteps, long ySteps)
{	
	int ret = xAxis_.SetRelativePosition((CAN29Long)xSteps, velocity_);
	if (ret != DEVICE_OK)
		return ret;

	ret = yAxis_.SetRelativePosition((CAN29Long)ySteps, velocity_);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}


int XYStageDevice::GetPositionSteps(long& xSteps, long& ySteps)
{
	int ret = xAxis_.GetPosition((CAN29Long&)xSteps);
	if (ret != DEVICE_OK)
		return ret;

	ret = yAxis_.GetPosition((CAN29Long&)ySteps);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int XYStageDevice::Home()
{
	int ret = xAxis_.FindLowerHardwareStop();
	if (ret != DEVICE_OK)
	return ret;

	ret = yAxis_.FindLowerHardwareStop();
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int XYStageDevice::Stop()
{
	int ret = xAxis_.Stop();
	if (ret != DEVICE_OK)
		return ret;

	ret = yAxis_.Stop();
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int XYStageDevice::SetOrigin()
{
	return SetAdapterOriginUm(0.0, 0.0);
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int XYStageDevice::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(can29_.port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(can29_.port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}

		//pProp->Get( port_);
		pProp->Get( can29_.port_);
		can29_.portInitialized_ = true;

	}

	return DEVICE_OK;
}



int XYStageDevice::OnTrajectoryVelocity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) 
	{
		// we are lazy and only check the x axis
		CAN29Long velocity;
		int ret = xAxis_.GetTrajectoryVelocity(velocity);
		if (ret != DEVICE_OK)
			return ret;
		pProp->Set( (float) (velocity/1000.0) );
	} 
	else if (eAct == MM::AfterSet) 
	{
		double tmp;
		pProp->Get(tmp);
		CAN29Long velocity = (CAN29Long) (tmp * 1000.0);
		int ret = xAxis_.SetTrajectoryVelocity(velocity);
		if (ret != DEVICE_OK)
			return ret;
		ret = yAxis_.SetTrajectoryVelocity(velocity);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;

}

int XYStageDevice::OnTrajectoryAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) 
	{
		// we are lazy and only check the x axis
		CAN29Long accel;
		int ret = xAxis_.GetTrajectoryAcceleration(accel);
		if (ret != DEVICE_OK)
			return ret;
		pProp->Set( (float) (accel / 1000.0) );
	} 
	else if (eAct == MM::AfterSet) 
	{
		double tmp;
		pProp->Get(tmp);
		CAN29Long accel = (long) (tmp * 1000.0);
		int ret = xAxis_.SetTrajectoryAcceleration(accel);
		if (ret != DEVICE_OK)
			return ret;
		ret = yAxis_.SetTrajectoryAcceleration(accel);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}


