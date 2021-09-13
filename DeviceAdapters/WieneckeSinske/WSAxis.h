///////////////////////////////////////////////////////////////////////////////
// FILE:          WSAxis.h
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
#ifndef _WSAXIS_H_
#define _WSAXIS_H_

#include "WS.h"
class WSAxisPollThread;

class WSAxis: public WSComponent
{
public:
	  WSAxis(WS* ws);
	  ~WSAxis();

	  int Initialize();
	  int UnInitialize();

	  int ReceiveMessageHandler(std::string& msg);

	  bool IsBusy() {return isBusy_;};

	  int GetPresent(bool& present);
	  int GetPositionCmd(int& position);
	  int GetPosition(int& position)
	  { 
		  position = actPosition_;
		  return DEVICE_OK;
	  };

	  int SetPosition(int position);
	  int SetRelativePosition(int position);
	  int Stop();

	  int GetLowerHardwareStop(int& position);
	  int GetUpperHardwareStop(int& position);
	  int FindLowerHardwareStop();
	  int FindUpperHardwareStop();

	  bool IsMoving();
	  int DoPoll();

private:
    WSAxisPollThread* pollThread_;

	int actPosition_;
	bool isBusy_;
};


class WSAxisPollThread : public MMDeviceThreadBase
{
public:
	WSAxisPollThread(WSAxis* wsAxis); 
	~WSAxisPollThread(); 
	int svc();
	int open (void*) { return 0;}
	int close(unsigned long) {return 0;}

	void Start();
	void Stop() {stop_ = true;}

private:
	WSAxis* wsAxis_;
	bool stop_;
	long intervalUs_;
	bool debug_;

	WSAxisPollThread& operator=(WSAxisPollThread& /*rhs*/) {assert(false); return *this;}
};

#endif // _WSAXIS_H_
