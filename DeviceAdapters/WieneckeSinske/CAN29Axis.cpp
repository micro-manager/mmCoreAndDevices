///////////////////////////////////////////////////////////////////////////////
// FILE:          CAN29Axis.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Zeiss CAN29 axis
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

#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "CAN29Axis.h"
#include <sstream>


///////////////////////////////////////////////////////////////////////////////

using namespace std;




///////////////////////////////////////////////////////////////////////////////
// CAN29Axis
//
int CAN29Axis::Initialize()
{
	int ret = StartMonitoring();
	if(ret != DEVICE_OK)
		return ret;

	// get initial positions
	ret = GetStatusCmd(actStatus_);
	if(ret != DEVICE_OK)
		return ret;

	ret = GetPositionCmd(actPosition_);
	if(ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int CAN29Axis::UnInitialize()
{
	StopMonitoring();
	Unlock();
	return DEVICE_OK;
}


int CAN29Axis::ReceiveMessageHandler(Message& msg)
{
	if(    (msg.CanSrc == canAddress_)								// from my CAN address?
		&& (msg.CmdNr == CMDNR_AXIS)								// with my command number
		&& (msg.Data.size() > 1 ) && (msg.Data[0] == devID_))		// with my deviceID
	{
		switch(msg.CmdCls)
		{
		case 0x07:   // events
			switch(msg.SubID)
			{
			case 0x01:		    // status
				MessageTools::GetULong(msg.Data, 1, actStatus_);
				isBusy_ = (actStatus_& 0x04) > 0;
				break;
			case 0x02:			// position
				MessageTools::GetLong(msg.Data, 1, actPosition_);
				break;
			}
			break;

		case 0x08:	 // answers
			break;

		case 0x09:   // completions
			switch(msg.SubID)
			{
			case 0x02:			// set position
			case 0x03:			// set relative position
				MessageTools::GetLong(msg.Data, 2, actPosition_);
				isBusy_ = false;
				break;

			case 0x18:			// find upper hardware stop
			case 0x19:			// find lower hardware stop
				Unlock();
				isBusy_ = false;
				break;
			}
			break;
		}
	}

	return DEVICE_OK;
}



/*
* Gets the application name
*/
int CAN29Axis::GetApplicationName(std::string& applName)
{	
	unsigned char dta[] = {0};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_SYSTEM, PROCID, 0x09, dta, 0), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetString(answer.Data, 0, applName);
	return ret;
}

/*
* Gets the status
*/
int CAN29Axis::GetStatusCmd(CAN29ULong& status)
{	
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x01, dta, 0), answer);  
	if(ret == DEVICE_OK)
	{
		ret =  MessageTools::GetULong(answer.Data, 1, status);
		isBusy_ = (status& 0x04) > 0;
	}
	return ret;
}


/*
* Checks if axis is present, by checking application name and status
*/
int CAN29Axis::GetPresent(bool& present, string expectedApplName)
{
	present = false;

	std::string name;
	int ret = GetApplicationName(name);
	if(ret != DEVICE_OK)
		return ret;

	CAN29ULong status;
	ret = GetStatus(status);
	if(ret != DEVICE_OK)
		return ret;

	// correct device and  status = "Device motorized" ? 
	present = (name == expectedApplName) &&  ((status & 0x4000) > 0);
 
	return ret;
}


/*
* GetPosition in nm (steps)
*/
int CAN29Axis::GetPositionCmd(CAN29Long& position)
{
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x02, dta, sizeof(dta)), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetLong(answer.Data, 1, position);

	return ret;
}


/*
* SetPosition in nm (steps)
*/
int CAN29Axis::SetPosition(CAN29Long position, CAN29Byte movemode)
{	
	unsigned char dta[2+CAN29LongSize] = {devID_, (unsigned char)movemode};   
	long tmp = htonl(position);
	memcpy(dta+2, &tmp, CAN29LongSize); 

	int ret = can29_->Send(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x02, dta, sizeof(dta)));  
	if(ret != DEVICE_OK)
		return ret;

	isBusy_ = true;
	return ret;
}

/*
* SetRelativePosition in nm (steps)
*/
int CAN29Axis::SetRelativePosition(CAN29Long position, CAN29Byte movemode)
{	
	unsigned char dta[2+CAN29LongSize] = {devID_, (unsigned char)movemode};   
	long tmp = htonl(position);
	memcpy(dta+2, &tmp, CAN29LongSize); 

	int ret = can29_->Send(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x03, dta, sizeof(dta)));  
	if(ret != DEVICE_OK)
		return ret;

	isBusy_ = true;
	return ret;
}


/*
* Stops all movements
*/
int CAN29Axis::Stop()
{	
	unsigned char dta[] = {devID_, (unsigned char)moveMode_};   

	return can29_->Send(Message(canAddress_, CAN_PC,  0x1B, CMDNR_AXIS, PROCID, 0x05, dta, sizeof(dta)));  
}

/*
* Locks the component
*/
int CAN29Axis::Lock()
{	
	unsigned char dta[] = {devID_, 1};   

	return can29_->Send(Message(canAddress_, CAN_PC,  0x1B, CMDNR_AXIS, PROCID, 0x61, dta, sizeof(dta)));  
}

/*
* Unlocks the component
*/
int CAN29Axis::Unlock()
{	
	unsigned char dta[] = {devID_, 0};   

	return can29_->Send(Message(canAddress_, CAN_PC,  0x1B, CMDNR_AXIS, PROCID, 0x61, dta, sizeof(dta)));  
}




/*
* Get position of lower hardware stop in nm (steps)
*/
int CAN29Axis::GetLowerHardwareStop(CAN29Long& position)
{
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x19, dta, sizeof(dta)), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetLong(answer.Data, 1, position);

	return ret;
}

/*
* Get position of upper hardware stop in nm (steps)
*/
int CAN29Axis::GetUpperHardwareStop(CAN29Long& position)
{
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x18, dta, sizeof(dta)), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetLong(answer.Data, 1, position);

	return ret;
}

/*
* Finds position of lower hardware stop
*/
int CAN29Axis::FindLowerHardwareStop()
{
	// lock component
	int ret = Lock();
	if(ret != DEVICE_OK)
		return ret;

	// find hardware stop
	unsigned char dta[] = {devID_};   
	ret = can29_->Send(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x19, dta, sizeof(dta)));  
	if(ret != DEVICE_OK)
		return ret;

	// set busy manually
	isBusy_ = true;
	return ret;
}

/*
* Finds position of upper hardware stop
*/
int CAN29Axis::FindUpperHardwareStop()
{
	// lock component
	int ret = Lock();
	if(ret != DEVICE_OK)
		return ret;

	// find hardware stop
	unsigned char dta[] = {devID_};   
	ret = can29_->Send(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x18, dta, sizeof(dta)));  
	if(ret != DEVICE_OK)
		return ret;

	// set busy manually
	isBusy_ = true;
	return ret;
}


/*
* Starts position and status monitoring
*/
int CAN29Axis::StartMonitoring()
{
	unsigned char dta[] = {devID_, 0x12, 0x00, 100, CAN_PC, 0xBB};   

	return can29_->Send(Message(canAddress_, CAN_PC,  0x1B, CMDNR_AXIS, PROCID, 0x1F, dta, sizeof(dta)));  

}

/*
* Stops position and status monitoring
*/
int CAN29Axis::StopMonitoring()
{
	unsigned char dta[] = {devID_, 0x00, 0x00, 100, CAN_PC, 0xBB};   

	return can29_->Send(Message(canAddress_, CAN_PC,  0x1B, CMDNR_AXIS, PROCID, 0x1F, dta, sizeof(dta)));  
}

/*
* Sets velocity for position moves in nm/s 
*/
int CAN29Axis::SetTrajectoryVelocity(CAN29Long velocity)
{
	unsigned char dta[1+CAN29LongSize] = {devID_,};   
	long tmp = htonl(velocity);
	memcpy(dta+1, &tmp, CAN29LongSize); 

	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x21, dta, sizeof(dta)), answer);  
	return ret;
}

/*
* Sets acceleration for position moves in nm/s² 
*/
int CAN29Axis::SetTrajectoryAcceleration(CAN29Long acceleration)
{
	unsigned char dta[1+CAN29LongSize] = {devID_,};   
	long tmp = htonl(acceleration);
	memcpy(dta+1, &tmp, CAN29LongSize); 

	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x19, CMDNR_AXIS, PROCID, 0x22, dta, sizeof(dta)), answer);  
	return ret;
}

/*
* Gets velocity for position moves in nm/s 
*/
int CAN29Axis::GetTrajectoryVelocity(CAN29Long& velocity)
{
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x21, dta, sizeof(dta)), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetLong(answer.Data, 1, velocity);

	return ret;
}

/*
* Gets acceleration for position moves in nm/s² 
*/
int CAN29Axis::GetTrajectoryAcceleration(CAN29Long& acceleration)
{
	unsigned char dta[] = {devID_};   
	Message answer;
	int ret = can29_->SendRead(Message(canAddress_, CAN_PC,  0x18, CMDNR_AXIS, PROCID, 0x22, dta, sizeof(dta)), answer);  
	if(ret == DEVICE_OK)
		ret = MessageTools::GetLong(answer.Data, 1, acceleration);

	return ret;
}


