///////////////////////////////////////////////////////////////////////////////
// FILE:          WS.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske protocol communication
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
#ifndef _WS_H_
#define _WS_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>
#include <queue>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_POSITION         10002
#define ERR_INVALID_SPEED            10003
#define ERR_PORT_CHANGE_FORBIDDEN    10004                                   
#define ERR_SET_POSITION_FAILED      10005                                   
#define ERR_INVALID_STEP_SIZE        10006                                   
#define ERR_LOW_LEVEL_MODE_FAILED    10007                                   
#define ERR_INVALID_MODE             10008 
#define ERR_DEVICE_NOT_ACTIVE        10012 
#define ERR_MODULE_NOT_FOUND         10014
#define ERR_TIMEOUT                  10021
#define ERR_INVALID_MESSAGE_DATA     10022






static const int WS_RCV_BUF_LENGTH = 1024;


class WSReceiveThread;
class WS;


/*
Base class for WS components
*/
class WSComponent 
{
public: 
	WSComponent(WS* ws);
	~WSComponent();

	virtual int ReceiveMessageHandler(std::string& msg) = 0;

	WS* ws_;
};


/*
Class containing WS message tools
*/
class WSMessageTools
{
public: 
	static int GetNumber(std::string& msg, int& result);
};



/*
Class for WS message IO
*/
class WS
{
public:
	std::string port_;
	bool portInitialized_;
	MM::Device* device_;
	MM::Core* core_;


	WS();
	~WS();

	int Initialize(MM::Device* device, MM::Core* core);
	int Send(std::string msg);
	int SendRead(std::string msg, std::string& answer, int timeoutMilliSec = 1000);
	int Receive(std::string msg);

	int AddReceiveMessageHandler(WSComponent* component);
	int RemoveReceiveMessageHandler(WSComponent* component);

private:
	WSReceiveThread* receiveThread_;
	std::string sendReadMessage_;
	std::string sendReadAnswer_;
	bool hasSendReadAnswer_;

	bool IsAnswer(std::string& question, std::string& answer);
	int ClearPort();
	int AppendByte(std::vector<unsigned char>& command, int& nextIndex, unsigned char byte);

	std::vector<WSComponent*> receiveMessageCallbackClasses_;
};


/*
* WSMessageParser: Takes a stream containing WS messages and
* splits this stream into individual messages.
*/
class WSMessageParser{
public:
	WSMessageParser(unsigned char* inputStream, long inputStreamLength);
	~WSMessageParser(){};
	int GetNextMessage(unsigned char* nextMessage, int& nextMessageLength);
	static const int messageMaxLength_ = 64;

private:
	unsigned char* inputStream_;
	long inputStreamLength_;
	long index_;
};


class WSReceiveThread : public MMDeviceThreadBase
{
public:
	WSReceiveThread(WS* ws); 
	~WSReceiveThread(); 
	int svc();
	int open (void*) { return 0;}
	int close(unsigned long) {return 0;}

	void Start();
	void Stop() {stop_ = true;}

private:
	WS* ws_;
	void interpretMessage(unsigned char* message);
	bool stop_;
	long intervalUs_;
	bool debug_;

	WSReceiveThread& operator=(WSReceiveThread& /*rhs*/) {assert(false); return *this;}
};



#endif // _WS_H_
