///////////////////////////////////////////////////////////////////////////////
// FILE:          WSAxis.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke Sinske axis 
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

#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "WSAxis.h"
#include <sstream>


///////////////////////////////////////////////////////////////////////////////

using namespace std;


WSAxis::WSAxis(WS* ws):
		WSComponent(ws),
		pollThread_(0),
		isBusy_(false),
		actPosition_(0)
{
}

WSAxis::~WSAxis()
{
	if (pollThread_ != 0)
		delete(pollThread_);
}


///////////////////////////////////////////////////////////////////////////////
// WSAxis
//
int WSAxis::Initialize()
{
	// get initial positions
	int ret = GetPositionCmd(actPosition_);
	if(ret != DEVICE_OK)
		return ret;

	pollThread_ = new WSAxisPollThread(this);
	pollThread_->Start();

	return DEVICE_OK;
}

int WSAxis::UnInitialize()
{
	return DEVICE_OK;
}


int WSAxis::ReceiveMessageHandler(std::string& msg)
{
	return DEVICE_OK;
}



/*
* Checks if axis is present, by checking communication
*/
int WSAxis::GetPresent(bool& present)
{
	present = false;

	int pos;
	int ret = GetPositionCmd(pos);
	if(ret != DEVICE_OK)
		return ret;
	
	present = true;
 	return ret;
}


/*
* GetPosition in nm (steps)
*/
int WSAxis::GetPositionCmd(int& position)
{
	std::string answer;
	int ret = ws_->SendRead("[3=PA?]", answer);  
	if(ret == DEVICE_OK)
		ret = WSMessageTools::GetNumber(answer, position);
	return ret;
}


/*
* SetPosition in nm (steps)
*/
int WSAxis::SetPosition(int position)
{	
	int ret = ws_->Send("[3=MA!" + std::to_string((long long)position) + "]");  
	if(ret != DEVICE_OK)
		return ret;

	isBusy_ = true;
	return ret;
}

/*
* SetRelativePosition in nm (steps)
*/
int WSAxis::SetRelativePosition(int position)
{	
	int ret = ws_->Send("[3=MR!" + std::to_string((long long)position) + "]");  
	if(ret != DEVICE_OK)
		return ret;

	isBusy_ = true;
	return ret;
}


/*
* Stops all movements
*/
int WSAxis::Stop()
{	
	return ws_->Send("[3=BR!]");  
}



/*
* Get position of lower hardware stop in nm (steps)
*/
int WSAxis::GetLowerHardwareStop(int& position)
{
	// get start position
	int startPos;
	int ret = GetPosition(startPos);
	if(ret == DEVICE_OK)
		return ret;

	// goto HW stop
	ret = FindLowerHardwareStop();
	if(ret == DEVICE_OK)
		return ret;

	// get HW stop position
	ret = GetPosition(position);
	if(ret == DEVICE_OK)
		return ret;

	// move back to start position
	ret = SetPosition(startPos);
	return ret;
}

/*
* Get position of upper hardware stop in nm (steps)
*/
int WSAxis::GetUpperHardwareStop(int& position)
{
	// get start position
	int startPos;
	int ret = GetPosition(startPos);
	if(ret == DEVICE_OK)
		return ret;

	// goto HW stop
	ret = FindUpperHardwareStop();
	if(ret == DEVICE_OK)
		return ret;

	// get HW stop position
	ret = GetPosition(position);
	if(ret == DEVICE_OK)
		return ret;

	// move back to start position
	ret = SetPosition(startPos);
	return ret;
}

/*
* Finds position of lower hardware stop
*/
int WSAxis::FindLowerHardwareStop()
{
	// find hardware stop
	int ret = ws_->Send("[3=HL!]");  
	if(ret != DEVICE_OK)
		return ret;

	// set busy manually
	isBusy_ = true;
	return ret;
}

/*
* Finds position of upper hardware stop
*/
int WSAxis::FindUpperHardwareStop()
{
	// find hardware stop
	int ret = ws_->Send("[3=HU!]");  
	if(ret != DEVICE_OK)
		return ret;

	// set busy manually
	isBusy_ = true;
	return ret;
}

/*
* Checks whether the axis is moving and updates the position
*/
bool WSAxis::IsMoving()
{
	// use poll command to detect movement
	std::string answer;
	int ret = ws_->SendRead("[3=PO?]", answer);  
	if(ret != DEVICE_OK)
	{
		ws_->core_->LogMessage(ws_->device_, "Error on Polling!", true);
		return false;
	}

	int result;
	WSMessageTools::GetNumber(answer, result);
	
	// update the position, if no more busy 
	if(result == 0)
		GetPositionCmd(actPosition_);
	
	isBusy_ = result == 1;
	return isBusy_;
}

/*
* Function called in the polling thread.
*/
int WSAxis::DoPoll()
{
	if(isBusy_)
		return DEVICE_OK;
	else 
		return GetPositionCmd(actPosition_);
}


///////////////////////////////////////////////////////////////////////////////
// WSAxisPollThread
//
// Thread that continuously polls the WS axis.
//
WSAxisPollThread::WSAxisPollThread(WSAxis* wsAxis) :
	wsAxis_ (wsAxis),  
	stop_ (true),
	debug_(true),
	intervalUs_(200000) // check every 200 ms for new states 
{  
}

WSAxisPollThread::~WSAxisPollThread()
{
	Stop();
	wait();
	wsAxis_->ws_->core_->LogMessage(wsAxis_->ws_->device_, "Destructing WSAxisPollThread", true);
	
}

int WSAxisPollThread::svc() {

	wsAxis_->ws_->core_->LogMessage(wsAxis_->ws_->device_, "Starting WSReceiveThread", true);
	while (!stop_) 
	{
		wsAxis_->DoPoll();
     	CDeviceUtils::SleepMs(intervalUs_/1000);
	}
	wsAxis_->ws_->core_->LogMessage(wsAxis_->ws_->device_, "WSReceiveThread finished", true);
	return 0;
}

void WSAxisPollThread::Start()
{

	stop_ = false;
	activate();
}

