#pragma once

///////////////////////////////////////////////////////////////////////////////
// FILE:          ummhUsb.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implementation of the universal hardware hub
//                that uses a serial port for communication
//                
// COPYRIGHT:     Artem Melnykov, 2022
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Artem Melnykov, melnykov.artem at gmail.com, 2021
// 

#ifndef _UMMHUSB_H_
#define _UMMHUSB_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"

#include "../UniversalMMHubSerial/ummhbase.h"

using namespace std;

struct mmmethoddescription {
	string method;
	string command;
};


struct mmpropertydescription {
	string name;

	bool isReadOnly;
	bool isPreini;

	bool isAction;
	string cmdAction;

	MM::PropertyType type;

	string valueString;
	vector<string> allowedValues;

	int valueInteger;
	int lowerLimitInteger;
	int upperLimitInteger;

	float valueFloat;
	float lowerLimitFloat;
	float upperLimitFloat;
};

struct mmdevicedescription {
	bool isValid;
	string reasonWhyInvalid;
	string type;
	string name;
	string description;
	MM::MMTime timeout;
	vector<mmmethoddescription> methods;
	vector<mmpropertydescription> properties;
};

class UniHub : public HubBase<UniHub>
{
	friend class BusyThread;
public:
	UniHub();
	~UniHub();

	// flag for stopping BusyThread
	bool stopbusythread_;

	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	int DetectInstalledDevices();
	bool Busy();
	// this is a preinitialization setting
	int OnUSBDevice(MM::PropertyBase* pProp, MM::ActionType eAct);

	// get device index in deviceCescriptionList
	int GetDeviceIndexFromName(string devicename);
	// convert method name to the command sent to device
	string ConvertMethodToCommand(string deviceName, string methodName);
	// concatenate and send a serial command
	int MakeAndSendOutputCommand(string devicename, string command, vector<string> values);
	// send serial command
	int SendCommand(string cmd, unsigned int timeoutMS);
	// receive serial command
	int ReceiveAnswer(string& ans, unsigned int timeoutMS);

	int UsbReceiveBuffer(unsigned char endpoint, unsigned char* buff, int size, int &receivedBytes, int timeoutMS);

	// block until serial command is received, exit at timeout regardless
	int ReceiveAndWaitForAnswer(string& ans, MM::MMTime timeout);
	// report values to the device
	int ReportToDevice(string devicename, string command, vector<string> vals);
	// report arror for a device
	int ReportErrorForDevice(string devicename, string command, vector<string> vals, int errorcode);
	// write error code, description, and report error in the log
	int WriteError(string addonstr, int errorcode);
	// action interface for error reporting
	int OnError(MM::PropertyBase* pProp, MM::ActionType eAct);
	// report timeout
	int ReportTimeoutError(string devicename);
	// check the incoming command
	int CheckIncomingCommand(vector<string> vs);

	// set device busy status
	void SetBusy(string devicename, bool val);
	// get device timeout value
	MM::MMTime GetTimeout(string devicename);
	// set device timeout value
	void SetTimeout(string devicename, MM::MMTime val);
	// get the time stamp of the last command sent to device
	MM::MMTime GetLastCommandTime(string devicename);
	// set the time stamp of the last command sent to device
	void SetLastCommandTime(string devicename, MM::MMTime val);

private:
	// make a human readable unique USB device descriptor
	string MakeUniqueUSBDescriptor(char* strDesc, unsigned char serialNumber);
	// communicate with the hardware and populate deviceDescriptionList
	int PopulateDeviceDescriptionList();
	// convert vector<string> to mmdevicedescription
	mmdevicedescription VectorstrToDeviceDescription(vector<string> vs);
	// get the string with device type from deviceDescriptionList based on device name
	string GetDeviceTypeFromName(string devicename);
	// thread-locked serial communication
	int UsbCommunication(char inorout, string cmd, string& ans, unsigned int timeoutMS);

	static const unsigned char bulk_ep_cmd_out = 0x01;
	static const unsigned char bulk_ep_cmd_in = 0x81;
	//static const unsigned char bulk_ep_buff_out = 0x02;
	//static const unsigned char bulk_ep_buff_in = 0x82;
	
	bool busy_;
	bool initialized_;
	int error_;
	std::string usbDeviceName_;
	MMThreadLock executeLock_;

	// thread for responding to communications from controller
	BusyThread* thr_;
};

class BusyThread : public MMDeviceThreadBase
{
	friend class UniHub;
public:
	BusyThread(UniHub* p);
	~BusyThread();
private:
	UniHub* pHub_;
	
	// A function that runs on a separate thread and serves to
	//  i) receive communication from the hardware via serial port;
	// ii) report received values to the devices;
	int svc(void);
};

class UmmhShutter : public UmmhDeviceUtilities, public CShutterBase<UmmhShutter>
{
public:
	UmmhShutter(const char* name);
	~UmmhShutter();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// Shutter API
	int SetOpen(bool open);
	int GetOpen(bool& open);
	int Fire(double deltaT);
	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	string name_;
	bool initialized_;
	UniHub* pHub_;

	bool open_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class UmmhStateDevice : public UmmhDeviceUtilities, public CStateDeviceBase<UmmhStateDevice>
{
public:
	UmmhStateDevice(const char* name);
	~UmmhStateDevice();

	// MMDevice API 
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// State API
	unsigned long GetNumberOfPositions() const;

	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	string name_;
	bool initialized_;
	UniHub* pHub_;

	unsigned long positionAkaState_;
	unsigned long numberOfPositions_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class UmmhStage : public UmmhDeviceUtilities, public CStageBase<UmmhStage>
{
public:
	UmmhStage(const char* name);
	~UmmhStage();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// Stage API
	int SetPositionUm(double pos);
	int GetPositionUm(double& pos);
	int Home();
	int Stop();

	int SetOrigin() { return DEVICE_UNSUPPORTED_COMMAND; }
	int SetPositionSteps(long steps)
	{
		double pos = steps * stepSize_um_;
		return SetPositionUm(pos);
	}
	int GetPositionSteps(long& steps)
	{
		steps = (long)(position_um_ / stepSize_um_);
		return DEVICE_OK;
	}
	int GetLimits(double& lower, double& upper)
	{
		lower = lowerLimit_um_;
		upper = upperLimit_um_;
		return DEVICE_OK;
	}
	
	bool IsContinuousFocusDrive() const { return false; }
	int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; };

	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	string name_;
	bool initialized_;
	UniHub* pHub_;

	double position_um_;
	double stepSize_um_;
	double lowerLimit_um_;
	double upperLimit_um_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class UmmhXYStage : public UmmhDeviceUtilities, public CXYStageBase<UmmhXYStage>
{
public:
	UmmhXYStage(const char* name);
	~UmmhXYStage();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// XYStage API
	int SetPositionUm(double posX, double posY);
	int GetPositionUm(double& posX, double& posY);
	int Home();
	int Stop();

	int SetOrigin() { return DEVICE_UNSUPPORTED_COMMAND; }
	int SetPositionSteps(long stepsX, long stepsY)
	{
		double posX = stepsX * stepSizeX_um_;
		double posY = stepsY * stepSizeY_um_;
		return SetPositionUm(posX,posY);
	}
	int GetPositionSteps(long& stepsX, long& stepsY)
	{
		stepsX = (long)(positionX_um_ / stepSizeX_um_);
		stepsY = (long)(positionY_um_ / stepSizeY_um_);
		return DEVICE_OK;
	}
	int GetLimitsUm(double& lowerX, double& upperX, double& lowerY, double& upperY)
	{
		lowerX = lowerLimitX_um_;
		upperX = upperLimitX_um_;
		lowerY = lowerLimitY_um_;
		upperY = upperLimitY_um_;
		return DEVICE_OK;
	}
	int GetStepLimits(long& lowerX, long& upperX, long& lowerY, long& upperY)
	{
		lowerX = (long)(lowerLimitX_um_ / stepSizeX_um_);;
		upperX = (long)(upperLimitX_um_ / stepSizeX_um_);
		lowerY = (long)(lowerLimitY_um_ / stepSizeY_um_);
		upperY = (long)(upperLimitY_um_ / stepSizeY_um_);
		return DEVICE_OK;
	}
	double GetStepSizeXUm() {
		return stepSizeX_um_;
	}
	double GetStepSizeYUm() {
		return stepSizeY_um_;
	}

	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; };

	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	string name_;
	bool initialized_;
	UniHub* pHub_;

	double positionX_um_;
	double stepSizeX_um_;
	double lowerLimitX_um_;
	double upperLimitX_um_;
	double positionY_um_;
	double stepSizeY_um_;
	double lowerLimitY_um_;
	double upperLimitY_um_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class UmmhGeneric : public UmmhDeviceUtilities, public CGenericBase<UmmhGeneric>
{
public:
	UmmhGeneric(const char* name);
	~UmmhGeneric();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	string name_;
	bool initialized_;
	UniHub* pHub_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class UmmhCamera : public UmmhDeviceUtilities, public CCameraBase<UmmhCamera>
{
public:
	UmmhCamera(const char* name);
	~UmmhCamera();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();
	void GetName(char* pszName) const;
	bool Busy();

	// Camera API
	double GetExposure() const {return exposure_;};
	void SetExposure(double exp);
	unsigned GetImageWidth() const {return roiWidth_;}
	unsigned GetImageHeight() const {return roiHeight_;}
	unsigned GetImageBytesPerPixel() const {return bytesPerPixel_;}
	unsigned GetBitDepth() const {return bitDepth_;}
	long GetImageBufferSize() const {
		long size = bytesPerPixel_*roiWidth_*roiHeight_;
		LogMessage("BufferSize=",false);
		LogMessage(to_string((long long)size).c_str(),false);
		return size;
	}
	int SnapImage();
	const unsigned char* GetImageBuffer();
	int SetBinning(int bin);
	int GetBinning() const {return binning_;}
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int ClearROI();

	// sequence acquisition
	int PrepareSequenceAcqusition() { return DEVICE_OK; }
	int StartSequenceAcquisition(double interval);
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StopSequenceAcquisition();
	int InsertImage();
	int RunSequenceOnThread(MM::MMTime startTime);
	bool IsCapturing();
	void OnThreadExiting() throw();
	int IsExposureSequenceable(bool& seq) const { seq = false; return DEVICE_OK; }

	// additional API calls specific to this adapter
	int SetPixelType(const char* pixelType);
	int SetBitDepth(int bitDepth);
	int SetTransferTimeout(int transferTimeout);

	// action interface
	// ----------------
	int OnAction(MM::PropertyBase* pProp, MM::ActionType eAct);

private:

	friend class MySequenceThread;

	string name_;
	bool initialized_;
	UniHub* pHub_;

	unsigned int imageMaxWidth_;
	unsigned int imageMaxHeight_;
	unsigned int roiX_;
	unsigned int roiY_;
	unsigned int roiWidth_;
	unsigned int roiHeight_;
	unsigned int bytesPerPixel_;
	unsigned int nComponents_; // 1 for grey, 4 for RGB
	unsigned int bitDepth_;
	double exposure_;
	int binning_;
	unsigned int transferTimeout_;
	unsigned char bulk_ep_camera_out_;
	unsigned char bulk_ep_camera_in_;

	MM::MMTime sequenceStartTime_;
	long imageCounter_;
	bool stopOnOverflow_;
	MySequenceThread* thd_;
	MMThreadLock imgPixelsLock_;

	int CreatePropertyBasedOnDescription(mmpropertydescription);
};

class MySequenceThread : public MMDeviceThreadBase
{
	friend class UmmhCamera;
	enum { default_numImages = 1, default_intervalMS = 100 };
public:
	MySequenceThread(UmmhCamera* pCam);
	~MySequenceThread();
	void Stop();
	void Start(long numImages, double intervalMs);
	bool IsStopped();
	void Suspend();
	bool IsSuspended();
	void Resume();
	double GetIntervalMs() { return intervalMs_; }
	void SetLength(long images) { numImages_ = images; }
	long GetLength() const { return numImages_; }
	long GetImageCounter() { return imageCounter_; }
	MM::MMTime GetStartTime() { return startTime_; }
	MM::MMTime GetActualDuration() { return actualDuration_; }
private:
	int svc(void) throw();
	double intervalMs_;
	long numImages_;
	long imageCounter_;
	bool stop_;
	bool suspend_;
	UmmhCamera* camera_;
	MM::MMTime startTime_;
	MM::MMTime actualDuration_;
	MM::MMTime lastFrameTime_;
	MMThreadLock stopLock_;
	MMThreadLock suspendLock_;
};


#endif //_UMMHUSB_H_