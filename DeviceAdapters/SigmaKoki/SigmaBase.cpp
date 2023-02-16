///////////////////////////////////////////////////////////////////////////////
// FILE:          SigmaBase.cpp
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#include "XYStage.h"
#include "ZStage.h"
#include "Shutter.h"
#include "SigmaBase.h"
#include "Camera.h"

/// <summary>
/// Constructor
/// </summary>
SigmaBase::SigmaBase(MM::Device *device):
	initialized_(false),
	port_("Undefined"),
	device_(device),
	core_(0)
{
}

/// <summary>
/// Destructor
/// </summary>
SigmaBase::~SigmaBase()
{
}



///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_XYStageDeviceName, MM::XYStageDevice, "XYStage (Advanced System)");
	RegisterDevice(g_ZStageDeviceName, MM::StageDevice, "ZStage (focusing actuator)");
	RegisterDevice(g_ShutterDeviceName_C2B1, MM::ShutterDevice, "SSH-C2B CH1");
	RegisterDevice(g_ShutterDeviceName_C2B2, MM::ShutterDevice, "SSH-C2B CH2");
	RegisterDevice(g_ShutterDeviceName_C4B1, MM::ShutterDevice, "SSH-C4B CH1");
	RegisterDevice(g_ShutterDeviceName_C4B2, MM::ShutterDevice, "SSH-C4B CH2");
	RegisterDevice(g_ShutterDeviceName_C4B3, MM::ShutterDevice, "SSH-C4B CH3");
	RegisterDevice(g_ShutterDeviceName_C4B4, MM::ShutterDevice, "SSH-C4B CH4");
	RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Camera");
}

/// <summary>
/// Device Creation 
/// </summary>
/// <param name="deviceName"></param>
/// <returns></returns>
MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0) 
	{
		return 0;
	}
	
	if (strcmp(deviceName, g_XYStageDeviceName) == 0)
	{
		XYStage* s = new XYStage();
		return s;
	}
	else if (strcmp(deviceName, g_ZStageDeviceName) == 0)
	{
		ZStage* s = new ZStage();
		return s;
		//return 0;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C2B1) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C2B1, 1);
		return s;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C2B2) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C2B2, 2);
		return s;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C4B1) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C4B1, 1);
		return s;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C4B2) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C4B2, 2);
		return s;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C4B3) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C4B3, 3);
		return s;
	}
	else if (strcmp(deviceName, g_ShutterDeviceName_C4B4) == 0)
	{
		Shutter* s = new Shutter(g_ShutterDeviceName_C4B4, 4);
		return s;
	}
	else if (strcmp(deviceName, g_CameraDeviceName) == 0)
	{
		Camera* s = new Camera();
		return s;
	}
	else 
	{
		return 0;
	}
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// Common API
///////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Clear port
/// </summary>
/// <returns></returns>
int SigmaBase::ClearPort() const
{
	core_->LogMessage(device_, "SigmaBase::ClearPort\n", true);
	//Clear contents of serial port 
	const int bufSize = 255;
	unsigned char clear[bufSize];
	unsigned long read = bufSize;
	int ret;
	while (read == (unsigned)bufSize)
	{
		ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufSize, read);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

/// <summary>
/// Send command string
/// </summary>
/// <param name="command"></param>
/// <returns></returns>
int SigmaBase::SendCommand(const std::string command) 
{
	core_->LogMessage(device_, ("SigmaBase::Send:" + command + "\n").c_str(), true);
	int ret = core_->SetSerialCommand(device_, port_.c_str(), command.c_str(), "\r\n");
	if (ret != DEVICE_OK) { return ret; }
	return DEVICE_OK;
}

/// <summary>
/// Recieve data
/// </summary>
/// <param name="data"></param>
/// <returns></returns>
int SigmaBase::RecieveData(std::string& data)
{
	const size_t BUFSIZE = 2048;
	char buf[BUFSIZE] = { '\0' };
	int ret = core_->GetSerialAnswer(device_, port_.c_str(), BUFSIZE, buf, "\r\n");
	if (ret != DEVICE_OK) { return ret; }
	data = buf;
	core_->LogMessage(device_, ("SigmaBase::Receive:" + data + "\n").c_str(), true);
	return DEVICE_OK;
}

/// <summary>
/// Send command and recieve data
/// </summary>
/// <param name="command"></param>
/// <param name="data"></param>
/// <returns></returns>
int SigmaBase::SendRecieve(const std::string command, std::string& data)
{
	int ret = SendCommand(command);
	if (ret != DEVICE_OK) { return ret; }
		
	ret = RecieveData(data);
	if (ret != DEVICE_OK){ return ret; }

	return DEVICE_OK;
}

/// <summary>
/// Send command and check recieve data 'OK'
/// </summary>
/// <param name="command"></param>
/// <returns></returns>
int SigmaBase::SendCheckRecievedOK(const std::string command)
{
	std::string data = "";
	int ret = SendRecieve(command, data);
	if (ret != DEVICE_OK){ return ret; }

	if (strcmp(data.c_str(), "OK") != 0)
	{
		return DEVICE_SERIAL_INVALID_RESPONSE;
	}
	return DEVICE_OK;
}

/// <summary>
/// Split by specified delimiter
/// </summary>
/// <param name="src">source string</param>
/// <param name="del">delimiter</param>
/// <returns></returns>
std::vector<std::string> SigmaBase::split(std::string src, char del)
{
	std::stringstream ss(src);
	vector<string> result;
	while (ss.good())
	{
		string substr;
		getline(ss, substr, del);
		result.push_back(substr);
	}
	return result;
}
