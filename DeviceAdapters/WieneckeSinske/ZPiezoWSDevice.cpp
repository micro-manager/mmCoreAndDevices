///////////////////////////////////////////////////////////////////////////////
// FILE:          ZPiezoWSDevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske ZPiezo Controller Driver for WS protocoll
//             
//
// AUTHOR:        S3L GmbH, info@s3l.de, www.s3l.de,  08/27/2021
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
#include "ZPiezoWSDevice.h"

#include <sstream>


///////////////////////////////////////////////////////////////////////////////
// ZPiezoWSDevice
//
ZPiezoWSDevice::ZPiezoWSDevice (): 
CStageBase<ZPiezoWSDevice>(),
	initialized_ (false),
	stepSize_um_(0.001),
	ws_(),
	zAxis_(&ws_)
{ 

	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_ZPiezoWSDeviceName, MM::String, true);

	// Description                                                            
	CreateProperty(MM::g_Keyword_Description, "Controller for z piezo stages", MM::String, true);

	// Port                                                                   
	CPropertyAction* pAct = new CPropertyAction (this, &ZPiezoWSDevice::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
	
	SetDelayMs(0);
}

ZPiezoWSDevice::~ZPiezoWSDevice() 
{
	zAxis_.UnInitialize();
	
	Shutdown();
}


bool ZPiezoWSDevice::Busy()
{	
	return zAxis_.IsMoving();
}

void ZPiezoWSDevice::GetName (char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_ZPiezoWSDeviceName);
}


int ZPiezoWSDevice::Initialize()
{
	if (!ws_.portInitialized_)
		return ERR_DEVICE_NOT_ACTIVE;

	ws_.Initialize(this, GetCoreCallback());

	zAxis_.Initialize();
	
	// check if this axis exists:
	bool presentZ;
	int ret = zAxis_.GetPresent(presentZ);
	if (ret != DEVICE_OK)
		return ret;
	if (!presentZ)
		return ERR_MODULE_NOT_FOUND;

	initialized_ = true;

	return DEVICE_OK;
}

int ZPiezoWSDevice::Shutdown()
{
	if (initialized_) initialized_ = false;
	return DEVICE_OK;
}

int ZPiezoWSDevice::GetLimits(double& lower, double& upper)
{	
	int zMinSteps, zMaxSteps;
	zAxis_.GetLowerHardwareStop((int&)zMinSteps);
	zAxis_.GetUpperHardwareStop((int&)zMaxSteps);

	lower = zMinSteps * stepSize_um_;
	upper = zMaxSteps * stepSize_um_;

	return DEVICE_OK;
}


int ZPiezoWSDevice::SetPositionUm(double pos)
{	
	return zAxis_.SetPosition((int)(1000*pos));
}

int ZPiezoWSDevice::SetRelativePositionUm(double pos)
{	
	return zAxis_.SetRelativePosition((int)(1000*pos));
}

int ZPiezoWSDevice::GetPositionUm(double& pos)
{	
	int steps;
	int ret = zAxis_.GetPosition((int&)steps);
	pos = steps * stepSize_um_;

	return ret;
}

int ZPiezoWSDevice::Home()
{
	int ret = zAxis_.FindLowerHardwareStop();
	return ret;
}

int ZPiezoWSDevice::Stop()
{
	int ret = zAxis_.Stop();
	return ret;
}

int ZPiezoWSDevice::SetOrigin()
{
	return DEVICE_OK;
	//return SetAdapterOriginUm(0.0);
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int ZPiezoWSDevice::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(ws_.port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(ws_.port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}

		//pProp->Get( port_);
		pProp->Get( ws_.port_);
		ws_.portInitialized_ = true;
	}

	return DEVICE_OK;
}





