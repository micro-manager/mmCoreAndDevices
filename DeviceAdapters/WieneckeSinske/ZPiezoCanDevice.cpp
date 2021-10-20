///////////////////////////////////////////////////////////////////////////////
// FILE:          ZPiezoCanDevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske ZPiezo Controller Driver for CAN protocoll
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

#ifdef WIN32
#include <windows.h>
#else
#include <arpa/inet.h>
#endif
#include "FixSnprintf.h"

#include "WieneckeSinske.h"
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "ZPiezoCANDevice.h"

#include <sstream>


///////////////////////////////////////////////////////////////////////////////
// ZPiezoCANDevice
//
ZPiezoCANDevice::ZPiezoCANDevice (): 
CStageBase<ZPiezoCANDevice>(),
	initialized_ (false),
	stepSize_um_(0.001),
	can29_(),
	zAxis_(CAN_ZPIEZOAXIS, 0, &can29_)
{ 

	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_ZPiezoCANDeviceName, MM::String, true);

	// Description                                                            
	CreateProperty(MM::g_Keyword_Description, "Controller for z piezo stages", MM::String, true);

	// Port                                                                   
	CPropertyAction* pAct = new CPropertyAction (this, &ZPiezoCANDevice::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

ZPiezoCANDevice::~ZPiezoCANDevice() 
{
	zAxis_.UnInitialize();
	
	Shutdown();
}


bool ZPiezoCANDevice::Busy()
{	
	return zAxis_.IsBusy();
}

void ZPiezoCANDevice::GetName (char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_ZPiezoCANDeviceName);
}


int ZPiezoCANDevice::Initialize()
{
	if (!can29_.portInitialized_)
		return ERR_DEVICE_NOT_ACTIVE;

	can29_.Initialize(this, GetCoreCallback());

	zAxis_.Initialize();
	
	// check if this axis exists:
	bool presentZ;
	int ret = zAxis_.GetPresent(presentZ, "WSB ZPiezo CAN");
	if (ret != DEVICE_OK)
		return ret;
	if (!presentZ)
		return ERR_MODULE_NOT_FOUND;

	initialized_ = true;

	return DEVICE_OK;
}

int ZPiezoCANDevice::Shutdown()
{
	if (initialized_) initialized_ = false;
	return DEVICE_OK;
}

int ZPiezoCANDevice::GetLimits(double& lower, double& upper)
{	
	long zMinSteps, zMaxSteps;
	zAxis_.GetLowerHardwareStop((CAN29Long&)zMinSteps);
	zAxis_.GetUpperHardwareStop((CAN29Long&)zMaxSteps);

	lower = zMinSteps * stepSize_um_;
	upper = zMaxSteps * stepSize_um_;

	return DEVICE_OK;
}


int ZPiezoCANDevice::SetPositionUm(double pos)
{	
	return zAxis_.SetPosition((CAN29Long)(1000*pos), 0);
}

int ZPiezoCANDevice::SetRelativePositionUm(double pos)
{	
	return zAxis_.SetRelativePosition((CAN29Long)(1000*pos), 0);
}

int ZPiezoCANDevice::GetPositionUm(double& pos)
{	
	CAN29Long steps;
	int ret = zAxis_.GetPosition((CAN29Long&)steps);
	pos = steps * stepSize_um_;

	return ret;
}

int ZPiezoCANDevice::Home()
{
	int ret = zAxis_.FindLowerHardwareStop();
	return ret;
}

int ZPiezoCANDevice::Stop()
{
	int ret = zAxis_.Stop();
	return ret;
}

int ZPiezoCANDevice::SetOrigin()
{
	return DEVICE_OK;
	//return SetAdapterOriginUm(0.0);
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int ZPiezoCANDevice::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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





