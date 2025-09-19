///////////////////////////////////////////////////////////////////////////////
// FILE:          LumeneraAce.h
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Lumenera  Cameras
//
// Copyright 2022 Photomics, Inc.
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
#include "ImgBuffer.h"
#include <iostream>

#include "Images/ImageDefinitions.h"
#include "lucamapi.h"
#include <memory>

namespace CameraInterface
{
	class Camera;

	namespace LucamAdapter
	{
		class LucamCamera;
	}
}

namespace Imaging
{
	class Image;
}

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
// Lumenera camera class
//////////////////////////////////////////////////////////////////////////////
//Callback class for putting frames in circular buffer as they arrive
//
//class CTempCameraEventHandler;
//class CircularBufferInserter;
class VideoSequenceThread;

class LumeneraCamera : public CLegacyCameraBase<LumeneraCamera> {

	friend class VideoSequenceThread;

public:
	LumeneraCamera();
	~LumeneraCamera();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;
	bool Busy() { return false; }


	// MMCamera API
	// ------------
	int SnapImage();
	const unsigned char* GetImageBuffer();

	unsigned  GetNumberOfComponents() const;
	unsigned GetImageWidth() const;
	unsigned GetImageHeight() const;
	unsigned GetImageBytesPerPixel() const;

	unsigned GetBitDepth() const;
	long GetImageBufferSize() const;
	double GetExposure() const;
	void SetExposure(double exp);
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int ClearROI();
	int GetBinning() const;
	int SetBinning(int binSize);


	int IsExposureSequenceable(bool& seq) const { seq = false; return DEVICE_OK; }


	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StartSequenceAcquisition(double interval_ms);
	int StopSequenceAcquisition();
	int PrepareSequenceAcqusition();
	bool IsCapturing();


	// action interface
	// ----------------
	int OnCameraIndex(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnHardwareTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerPin(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerPolarity(MM::PropertyBase* pProp, MM::ActionType eAct);


	int OnBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTapConfiguration(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDemosaicingMethod(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLightSource(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMirror(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSaturation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnContrast(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHighConversionGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCooling(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnIris(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAbsoluteFocus(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnReadOnlyProperty(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSingleProperty(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLinkedVideoAndStillProperty(MM::PropertyBase* pProp, MM::ActionType eAct, bool requiresStreamRefresh = false);

	int OnWhiteBalance(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);
	

private:

	VideoSequenceThread* sequenceThread_;

	bool colorCamera_;
	int cameraIndex_;
	CameraInterface::LucamAdapter::LucamCamera* camera_;
	bool hasSwitchingExposure_;
	double switchingExposure_;

	bool initialized_;

	ImgBuffer image_;
	unsigned int components_;
	Imaging::IMAGE_FORMAT format_;
	Imaging::IMAGE_BIT_DEPTH bitDepth_;

	MMThreadLock imageLock_;

	//MM::MMTime startTime_;


	//////internal functions
	bool hasCamera();
	CameraInterface::LucamAdapter::LucamCamera* camera() const;
	bool cameraRequiresClockThrottling(CameraInterface::Camera* cam);
	void throttleClockSpeed(CameraInterface::Camera* cam, const std::string& clock);

	bool initializeHardwareInterface();
	int initializeImageBuffer();
	int createProperties(CameraInterface::Camera* camera);

	int getBitDepthFromCamera(Imaging::IMAGE_BIT_DEPTH& bitDepth) const;
	int readCameraPropertyValue(const std::string& name, std::string& value) const;
	int writeCameraPropertyValue(const std::string& name, const std::string& value);
	unsigned getSensorWidth();
	unsigned getSensorHeight();
	int setCameraRoi(unsigned x, unsigned y, unsigned xSize, unsigned ySize, unsigned binningFactor);
	int getCameraRoi(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	void updateImageBuffer(std::unique_ptr<Imaging::Image>&& image);
	bool isFlippingEnabled();
	bool isMirrorEnabled();
	int refreshStream();
	int resizeImageBuffer();
	int setBitDepth(const Imaging::IMAGE_BIT_DEPTH& bitDepth);
	bool requiresStillStream(double value, const std::string& hardwareTrigger);
	bool requiresVideoStream(double value, const std::string& hardwareTrigger);
	bool isVideoStreamingMode();
	bool isStillStreamingMode();
	int applyStreamMode(double exposure, const std::string& hardwareTrigger);
	int getBinValueFromSampling(const std::string& sampling);
	std::vector<std::string> getBinningOptions();
	std::string getSamplingFromBinValue(int binValue);

	Imaging::IMAGE_BIT_DEPTH getBitDepthFromPixelType(const std::string& pixelType);
	Imaging::IMAGE_FORMAT getImageFormatFromPixelType(const std::string& pixelType);
	int createLinkedGainProperty(CameraInterface::Camera* camera, const std::string& propertyName, const char* uiName);
	void waitForCameraStream(bool streaming = true);
	int setCoolingState(const std::string& state);
	int createWhiteBalanceTargetProperty(CameraInterface::Camera* camera, const char* uiName);

	bool cameraSupportsProperty(const std::string& name);

	std::string getPropertyName(const std::string& name);
	std::string getVideoPropertyName(const std::string& name);
	std::string getStillPropertyName(const std::string& name);

	std::vector<std::string> getBitDepthOptions();

	int captureImage();

	int captureSequenceImage();
	void sequenceEnded() noexcept;
	bool hardwareTriggerEnabled();

	void setHardwareTriggeringEnabledState(bool state);
	void disableHardwareTriggering();
	void enableHardwareTriggering();

};

//Enumeration used for distinguishing different events.
enum TemperatureEvents
{
	TempCritical = 100,
	TempOverTemp = 200
};

// Number of images to be grabbed.
//static const uint32_t c_countOfImagesToGrab = 5;


//// Example handler for camera events.
//class CTempCameraEventHandler : public CLumeneraUniversalCameraEventHandler
//{
//private:
//	LumeneraCamera* dev_;
//public:
//	CTempCameraEventHandler(LumeneraCamera* dev);
//	virtual void OnCameraEvent(CLumeneraUniversalInstantCamera& camera, intptr_t userProvidedId, GenApi::INode* pNode);
//};
//
//
//class CircularBufferInserter : public CImageEventHandler {
//private:
//	LumeneraCamera* dev_;
//
//public:
//	CircularBufferInserter(LumeneraCamera* dev);
//
//	virtual void OnImageGrabbed(CInstantCamera& camera, const CGrabResultPtr& ptrGrabResult);
//};

