///////////////////////////////////////////////////////////////////////////////
// FILE:          WS.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske protcol communication
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

#include "WS.h"
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include <sstream>

#include <assert.h>


///////////////////////////////////////////////////////////////////////////////
// WSComponent
//
WSComponent::WSComponent(WS* ws):
	ws_(ws)
{
	ws_->AddReceiveMessageHandler(this);
}

WSComponent::~WSComponent()
{
	ws_->RemoveReceiveMessageHandler(this);
}




///////////////////////////////////////////////////////////////////////////////
// WSMessageTools:  Class containing WS message tools
//
int WSMessageTools::GetNumber(std::string& msg, int& result)
{
	result = 0;

	size_t startNumber = msg.find('=');
	if(startNumber == std::string::npos)
		return ERR_INVALID_MESSAGE_DATA;

	std::string tmp = "";
	for (unsigned int i=(unsigned int)startNumber+1; i < msg.length()-1; i++)
		tmp += msg[i];

	result = std::stoi(tmp, nullptr, 10);

	return DEVICE_OK;
}





///////////////////////////////////////////////////////////////////////////////
// WS
//
WS::WS():
	port_("Undefined"),
	portInitialized_(false),
	receiveThread_(0),
	hasSendReadAnswer_(false),
	sendReadAnswer_(),
	sendReadMessage_()
{	
}


WS::~WS()
{
	if (receiveThread_ != 0)
		delete(receiveThread_);
}



int WS::Initialize(MM::Device* device, MM::Core* core)
{
	device_ = device;
	core_ = core;

	ClearPort();

	receiveThread_ = new WSReceiveThread(this);
	receiveThread_->Start();


	return DEVICE_OK;
}


int WS::Send(std::string msg)
{
	// Prepare command according to WS Protocol
	std::vector<unsigned char> preparedCommand = std::vector<unsigned char>(msg.data(), msg.data() + msg.length());
	
	// send command
	int ret = core_->WriteToSerial(device_, port_.c_str(), &(preparedCommand[0]), (unsigned long)msg.length());
	if (ret != DEVICE_OK)                                                     
		return ret;                                                            

	return DEVICE_OK; 
}

int WS::SendRead(std::string msg, std::string& answer, int timeoutMilliSec)
{
	sendReadMessage_ = msg;
	hasSendReadAnswer_ = false;

	// send message out
	int res = Send(msg);
	if(res != DEVICE_OK)
		return res;

	// wait for answer
	MM::MMTime dTimeout = MM::MMTime (timeoutMilliSec*1000);
	MM::MMTime start = core_->GetCurrentMMTime();
	while(!hasSendReadAnswer_ && ((core_->GetCurrentMMTime() - start) < dTimeout)) 
	{
		CDeviceUtils::SleepMs(20);
	}
	if (!hasSendReadAnswer_)
		return ERR_TIMEOUT;

	// return answer
	answer = sendReadAnswer_;

	return DEVICE_OK;
}



int WS::Receive(std::string msg)
{
	// check if it is an expected answer for SendRead function
	if(IsAnswer(sendReadMessage_, msg))
	{
		sendReadAnswer_ = msg;
		hasSendReadAnswer_ = true;
	}

	// call all registered ReceiveMessageHandlers
	for(unsigned int i= 0; i< receiveMessageCallbackClasses_.size(); i++)
		receiveMessageCallbackClasses_[i]->ReceiveMessageHandler(msg);

	return DEVICE_OK;
}


bool WS::IsAnswer(std::string& question, std::string& answer)
{
	return (question.compare(0,3,answer,0,3) == 0);
}



/*
Appends a data byte to the WS command array. 
*/
int WS::AppendByte(std::vector<unsigned char>& command, int& nextIndex, unsigned char byte)
{
	// add data byte
	command[nextIndex++] = byte;

	return DEVICE_OK; 
}

int WS::ClearPort()
{
	// Clear contents of serial port 
	const unsigned int bufSize = 255;
	unsigned char clear[bufSize];
	unsigned long read = bufSize;
	int ret;
	while (read == bufSize)
	{
		ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufSize, read);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
} 


int WS::AddReceiveMessageHandler(WSComponent* component)
{
	receiveMessageCallbackClasses_.push_back(component);
	return DEVICE_OK;
}

int WS::RemoveReceiveMessageHandler(WSComponent* component)
{
	for(unsigned int i = 0; i< receiveMessageCallbackClasses_.size(); i++)
	{
		if(receiveMessageCallbackClasses_[i] == component)
		{
			receiveMessageCallbackClasses_.erase(receiveMessageCallbackClasses_.begin()+i);
			return DEVICE_OK;
		}
	}
	return DEVICE_OK;
}




///////////////////////////////////////////////////////////////////////////////
// WSMessageParser
//
//  Utility class for WSReceiveThread
//  Takes an input stream and returns WS messages in the GetNextMessage method
//
WSMessageParser::WSMessageParser(unsigned char* inputStream, long inputStreamLength) :
index_(0)
{
	inputStream_ = inputStream;
	inputStreamLength_ = inputStreamLength;
}

/*
* Find a message starting with '[' and ends with ']'.  
*/
int WSMessageParser::GetNextMessage(unsigned char* nextMessage, int& nextMessageLength) {
	bool startFound = false;
	bool endFound = false;

	nextMessageLength = 0;
	long remainder = index_;
	while ( (endFound == false) && (index_ < inputStreamLength_) && (nextMessageLength < messageMaxLength_) ) {
		if (inputStream_[index_] == '[') {
			startFound = true;
		}
		else if (inputStream_[index_] == ']' ) {
			endFound = true;
		}

		if (startFound) {
			nextMessage[nextMessageLength] = inputStream_[index_];
			nextMessageLength++;
		}
		index_++;
	}
	if (endFound)
	{
		nextMessage[nextMessageLength] = 0;
		nextMessageLength++;
		return 0;
	}
	else {
		// no more complete message found, return the whole stretch we were considering:
		for (long i = remainder; i < inputStreamLength_; i++)
			nextMessage[i-remainder] = inputStream_[i];
		nextMessageLength = inputStreamLength_ - remainder;
		return -1;
	}
}



///////////////////////////////////////////////////////////////////////////////
// WSReceiveThread
//
// Thread that continuously monitors messages from WS.
//
WSReceiveThread::WSReceiveThread(WS* ws) :
	ws_ (ws),  
	stop_ (true),
	debug_(true),
	intervalUs_(10000) // check every 10 ms for new messages, 
{  
}

WSReceiveThread::~WSReceiveThread()
{
	Stop();
	wait();
	ws_->core_->LogMessage(ws_->device_, "Destructing WSReceiveThread", true);
}

void WSReceiveThread::interpretMessage(unsigned char* message)
{
	std::string msg(reinterpret_cast<char*>(message));
    ws_->Receive(msg);
}

int WSReceiveThread::svc() {

	ws_->core_->LogMessage(ws_->device_, "Starting WSReceiveThread", true);

	unsigned long dataLength;
	unsigned long charsRead = 0;
	unsigned long charsRemaining = 0;
	unsigned char rcvBuf[WS_RCV_BUF_LENGTH];
	memset(rcvBuf, 0, WS_RCV_BUF_LENGTH);

	while (!stop_) 
	{
		do { 
			dataLength = WS_RCV_BUF_LENGTH - charsRemaining;
			int ret = ws_->core_->ReadFromSerial(ws_->device_, ws_->port_.c_str(), rcvBuf + charsRemaining, dataLength, charsRead); 

			if (ret != DEVICE_OK) 
			{
				std::ostringstream oss;
				oss << "WSReceiveThread: ERROR while reading from serial port, error code: " << ret;
				ws_->core_->LogMessage(ws_->device_, oss.str().c_str(), false);
			} 
			else if (charsRead > 0) 
			{
				WSMessageParser parser(rcvBuf, charsRead + charsRemaining);
				do 
				{
					unsigned char message[WS_RCV_BUF_LENGTH];
					int messageLength;
					ret = parser.GetNextMessage(message, messageLength);
					if (ret == 0) 
					{                  
						// Report 
						if (debug_) 
						{
							std::ostringstream os;
							os << "WSReceiveThread incoming message: ";
							for (int i=0; i< messageLength; i++) 
							{
								os << std::hex << (unsigned int)message[i] << " ";
							}
							ws_->core_->LogMessage(ws_->device_, os.str().c_str(), true);
						}
						// and do the real stuff
						interpretMessage(message);
					}
					else 
					{
						// no more messages, copy remaining (if any) back to beginning of buffer
						if (debug_ && messageLength > 0) 
						{
							std::ostringstream os;
							os << "WSReceiveThread no message found!: ";
							for (int i = 0; i < messageLength; i++) 
							{
								os << std::hex << (unsigned int)message[i] << " ";
								rcvBuf[i] = message[i];
							}
							ws_->core_->LogMessage(ws_->device_, os.str().c_str(), true);
						}
						memset(rcvBuf, 0, WS_RCV_BUF_LENGTH);
						for (int i = 0; i < messageLength; i++) 
						{
							rcvBuf[i] = message[i];
						}
						charsRemaining = messageLength;
					}
				} while (ret == 0);
			}
		} 
		while ((charsRead != 0) && (!stop_)); 
		CDeviceUtils::SleepMs(intervalUs_/1000);
	}
	ws_->core_->LogMessage(ws_->device_, "WSReceiveThread finished", true);
	return 0;
}

void WSReceiveThread::Start()
{

	stop_ = false;
	activate();
}

