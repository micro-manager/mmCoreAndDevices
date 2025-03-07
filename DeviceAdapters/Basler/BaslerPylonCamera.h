///////////////////////////////////////////////////////////////////////////////
// FILE:          BaslerAce.h
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Basler  Cameras
//
// Copyright 2018 Henry Pinkard
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this 
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other 
// materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without specific 
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
// DAMAGE.


#pragma once

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <string>
#include <vector>
#include <map>
#include "ImageMetadata.h"
#include <iostream>
#include <pylon/PylonIncludes.h>


//////////////////////////////////////////////////////////////////////////////
// Error codes
//
//#define ERR_UNKNOWN_BINNING_MODE 410
enum
{
	ERR_SERIAL_NUMBER_REQUIRED = 20001,
	ERR_SERIAL_NUMBER_NOT_FOUND,
	ERR_CANNOT_CONNECT,
};

//////////////////////////////////////////////////////////////////////////////
// Basler camera class
//////////////////////////////////////////////////////////////////////////////
//Callback class for putting frames in circular buffer as they arrive

class CMMCameraEventHandler;
class BufferInserter;
class BaslerCamera : public CNewAPICameraBase<BaslerCamera>  {
public:
	BaslerCamera();
	~BaslerCamera();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;      
	bool Busy() {return false;}
	

	// MMCamera API (shared between old and new API)
	// ------------
	
	unsigned GetNumberOfComponents() const;
	unsigned GetImageWidth() const;
	unsigned GetImageHeight() const;
	unsigned GetImageBytesPerPixel() const;

	unsigned GetBitDepth() const;
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize); 
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize); 
	int ClearROI();
	int GetBinning() const;
	int SetBinning(int binSize);
	bool IsCapturing();

	

	/////////////////////////////////////
	///////// New Camera API ///////////
	int TriggerSoftware();

	int AcquisitionArm(int frameCount);
	int AcquisitionArm();

	int AcquisitionStart();
	int AcquisitionStop();
	int AcquisitionAbort();

	////////////////////////
	///// End new Camera API
	////////////////////////


	// action interface
	// ----------------
	// Standard properties
	int OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerActivation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLineSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLineMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLineInverter(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLineSource(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLineStatus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerOverlap(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureTime(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnEventSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEventNotification(MM::PropertyBase* pProp, MM::ActionType eAct);


	// non-standard properties
	int OnAcqFramerate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAcqFramerateEnable(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoExpore(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinningMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDeviceLinkThroughputLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHeight(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnInterPacketDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLightSourcePreset(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnResultingFramerate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnReverseX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnReverseY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSensorReadoutMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnShutterMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperatureState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWidth(MM::PropertyBase* pProp, MM::ActionType eAct);




	// Convenience functions
	std::string NodeToString(const char* str) const;
 
	int HandleEnumerationProperty(MM::PropertyBase* pProp, MM::ActionType eAct, const char* nodeName);


	void ResizeSnapBuffer();
	void ReduceImageSize(int64_t Width, int64_t Height);
	int CheckForBinningMode(CPropertyAction *pAct);
	void AddToLog(std::string msg);
	void CopyToImageBuffer(CGrabResultPtr image);
	void UpdateTemperature();
	void RGBPackedtoRGB(void* destbuffer, const CGrabResultPtr& ptrGrabResult);

	
	//Genicam Callback
	void ResultingFramerateCallback(GenApi::INode* pNode);


	CImageFormatConverter *converter;
    BufferInserter *ImageHandler_;
	std::string EnumToString(EDeviceAccessiblityInfo DeviceAccessiblityInfo);
	void* Buffer4ContinuesShot;

	std::vector<std::string> GetAvailableEnumValues(const GenApi::IEnumeration& node);

	// Standard property initialization functions
	int InitTriggerSelectorStandardProperty();
	int InitTriggerModeStandardProperty();
	int InitTriggerSourceStandardProperty();
	int InitTriggerActivationStandardProperty();
	int InitTriggerDelayStandardProperty();
	int InitTriggerOverlapStandardProperty();
	int InitExposureModeStandardProperty();
	int InitExposureTimeStandardProperty();
	int InitLineSelectorStandardProperty();
	int InitLineModeStandardProperty();
	int InitLineInverterStandardProperty();
	int InitLineSourceStandardProperty();
	int InitLineStatusStandardProperty();
	int InitEventSelectorStandardProperty();
	int InitEventNotificationStandardProperty();

private:



	CBaslerUniversalInstantCamera *camera_;
	CMMCameraEventHandler *pEventHandler_;

    int nComponents_;
	unsigned bitDepth_;
	bool  colorCamera_;

	unsigned maxWidth_, maxHeight_;
	unsigned multiFrameAcqCount_;
	int64_t DeviceLinkThroughputLimit_;
	int64_t InterPacketDelay_;
	double ResultingFrameRatePrevious;
	double acqFramerate_, acqFramerateMax_, acqFramerateMin_;
	double exposure_us_, exposureMax_, exposureMin_;
	double gain_, gainMax_, gainMin_;
	double offset_, offsetMin_, offsetMax_;


	std::string binningFactor_;
	std::string pixelType_;
	std::string reverseX_, reverseY_;
	std::string sensorReadoutMode_;
	std::string setAcqFrm_;
	std::string shutterMode_;
	std::string temperature_;
	std::string temperatureState_;
	

	INodeMap* nodeMap_;

	bool initialized_;


	//MM::MMTime startTime_;

    int InitOrSyncTriggerSelectorStandardProperty();
    int InitOrSyncTriggerModeStandardProperty();
    int InitOrSyncTriggerSourceStandardProperty();
    int InitOrSyncTriggerActivationStandardProperty();
    int InitOrSyncTriggerDelayStandardProperty();
    int InitOrSyncTriggerOverlapStandardProperty();
    int InitOrSyncExposureModeStandardProperty();
    int InitOrSyncExposureTimeStandardProperty();
    int InitOrSyncLineSelectorStandardProperty();
    int InitOrSyncLineModeStandardProperty();
    int InitOrSyncLineInverterStandardProperty();
    int InitOrSyncLineSourceStandardProperty();
    int InitOrSyncLineStatusStandardProperty();
    int InitOrSyncEventSelectorStandardProperty();
    int InitOrSyncEventNotificationStandardProperty();

};

//Enumeration used for distinguishing different events.
enum TemperatureEvents
{
	TempCritical = 100,
	TempOverTemp = 200    
};

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 5;


// Example handler for camera events.
class CMMCameraEventHandler : public CBaslerUniversalCameraEventHandler
{
private:
	BaslerCamera* dev_;
public:
	CMMCameraEventHandler(BaslerCamera* dev);
	virtual void OnCameraEvent(CBaslerUniversalInstantCamera& camera, intptr_t userProvidedId, GenApi::INode* pNode);
};


class BufferInserter : public CImageEventHandler {
private:
	BaslerCamera* dev_;

public:
	BufferInserter(BaslerCamera* dev);

	virtual void OnImageGrabbed( CInstantCamera& camera, const CGrabResultPtr& ptrGrabResult);
};

