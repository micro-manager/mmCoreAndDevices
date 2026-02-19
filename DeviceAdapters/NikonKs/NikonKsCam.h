///////////////////////////////////////////////////////////////////////////////
// FILE:          NikonKsCam.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Nikon DS-Ri2 and DS-Qi2
//                Based on several other DeviceAdapters, 
//				  especially ThorLabsUSBCamera
//
// AUTHOR:        Andrew Gomella, andrewgomella@gmail.com, 2015
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef _NIKONKS_H_
#define _NIKONKS_H_

#include "DeviceBase.h"
#include "MMDevice.h"
#include "ImgBuffer.h"
#include "DeviceUtils.h"
#include "DeviceThreads.h"
#include "DeviceEvents.h"

#include "DsCam.h"
#include <DsCamCommand.h>
#include <DsCamEvent.h>
#include <DsCamFeature.h>
#include <DsCamImage.h>

#include <map>


//////////////////////////////////////////////////////////////////////////////
// Error codes
//

//////////////////////////////////////////////////////////////////////////////
// NikonKsCam class
//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;

class NikonKsCam : public CCameraBase<NikonKsCam>
{
public:
	NikonKsCam();
	~NikonKsCam();

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

	unsigned GetImageWidth() const{return img_.Width();};
	unsigned GetImageHeight() const{return img_.Height();};
	unsigned GetImageBytesPerPixel() const{return img_.Depth();};
	unsigned GetBitDepth() const{return bitDepth_;};
	long GetImageBufferSize() const{return img_.Width() * img_.Height() * GetImageBytesPerPixel();}
	int GetComponentName(unsigned comp, char* name);
	unsigned GetNumberOfComponents() const{return numComponents_;};
	double GetExposure() const;
	void SetExposure(double exp);
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int ClearROI();
	int StartSequenceAcquisition(double interval);
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StopSequenceAcquisition();
	int InsertImage();
	int ThreadRun();
	bool IsCapturing();
	void OnThreadExiting() throw();
	int GetBinning() const;
	int SetBinning(int bS);

	int IsExposureSequenceable(bool& isSequenceable) const
	{
		isSequenceable = false;
		return DEVICE_OK;
	}


	// action interface
	// ----------------
	int OnCameraSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnImageFormat(MM::PropertyBase*, MM::ActionType);
	int OnExposureTime(MM::PropertyBase*, MM::ActionType);
	int OnHardwareGain(MM::PropertyBase*, MM::ActionType);
	int OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSharpness(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSaturation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWhiteBalanceRed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWhiteBalanceBlue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPresets(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureTimeLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGainLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMeteringAreaLeft(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMeteringAreaTop(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMeteringAreaWidth(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMeteringAreaHeight(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureBias(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMeteringMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCaptureMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSignalExposureEnd(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSignalTriggerReady(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSignalDeviceCapture(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureOutput(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRoiX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRoiY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerFrame(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnList(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId);
	int OnRange(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId);
	int OnExposureChange(MM::PropertyBase* pProp, MM::ActionType eAct, lx_uint32 uiFeatureId);
	
	/* KsCam Event Handler must be public */
	void DoEvent(const lx_uint32 eventCameraHandle, CAM_Event* pEvent, void* pTransData);

private:
	int CreateKsProperty(lx_uint32 FeatureId, CPropertyAction* pAct);
	void SearchDevices();
	void Bgr8ToBGRA8(unsigned char* dest, unsigned char* src, lx_uint32 width, lx_uint32 height);
	void GrabFrame();
	void SetFeature(lx_uint32 uiFeatureId);
	void GetAllFeaturesDesc();
	void GetAllFeatures();
	void UpdateImageSettings();
	void SetROILimits();
	void SetMeteringAreaLimits();
	void Command(const lx_wchar* wszCommand);
	const char* ConvFeatureIdToName(const lx_uint32 uiFeatureId);

	//  Device Info ----------------------------------------
	BOOL isOpened_;
	BOOL isInitialized_;
	BOOL isRi2_;
	lx_uint32 deviceIndex_;
	lx_uint32 deviceCount_;
	CAM_Device device_;
	CAM_Image image_;

	//  Camera ---------------------------------------------
	lx_uint32 cameraHandle_;
	CAM_CMD_GetFrameSize frameSize_;
	char camID_[CAM_NAME_MAX + CAM_VERSION_MAX];

	//  Callback -------------------------------------------
	void* ptrEventData_;

	// Feature ---------------------------------------------
	Vector_CAM_FeatureValue vectFeatureValue_;
	CAM_FeatureDesc* featureDesc_;
	//CAM_FeatureDescFormat m_stDescFormat;
	std::map<lx_uint32, lx_uint32> mapFeatureIndex_;

	inline void Free_Vector_CAM_FeatureValue(Vector_CAM_FeatureValue& vectFeatureValue)
	{
		if (vectFeatureValue.pstFeatureValue != NULL)
		{
			delete [] vectFeatureValue.pstFeatureValue;
			ZeroMemory(&vectFeatureValue, sizeof(Vector_CAM_FeatureValue));
		}
	}
	
	//Image info
	ImgBuffer img_;
	long imageWidth_;
	long imageHeight_;
	int bitDepth_;
	int byteDepth_;
	int numComponents_;
	bool color_;

	MMEvent frameDoneEvent_; // Signals the sequence thread when a frame was captured
	bool busy_;
	bool stopOnOverFlow_;
	MM::MMTime readoutStartTime_;
	MM::MMTime sequenceStartTime_;
	unsigned roiX_;
	unsigned roiY_;
	unsigned roiWidth_;
	unsigned roiHeight_;
	long imageCounter_;
	long binSize_;
	double readoutUs_;
	volatile double framesPerSecond_;

	MMThreadLock imgPixelsLock_;
	friend class MySequenceThread;
	MySequenceThread* thd_;
	char* cameraBuf_; // camera buffer for image transfer
	int cameraBufId_; // buffer id, required by the SDK
};

class MySequenceThread : public MMDeviceThreadBase
{
	friend class NikonKsCam;

	enum
	{
		default_numImages=1,
		default_intervalMS = 100
	};

public:
	MySequenceThread(NikonKsCam* pCam);
	~MySequenceThread();
	void Stop();
	void Start(long numImages, double intervalMs);
	bool IsStopped();
	void Suspend();
	bool IsSuspended();
	void Resume();

	double GetIntervalMs()
	{
		return intervalMs_;
	}

	void SetLength(long images)
	{
		numImages_ = images;
	}

	long GetLength() const
	{
		return numImages_;
	}

	long GetImageCounter()
	{
		return imageCounter_;
	}

	MM::MMTime GetStartTime()
	{
		return startTime_;
	}

	MM::MMTime GetActualDuration()
	{
		return actualDuration_;
	}

private:
	int svc(void) throw();
	bool stop_;
	bool suspend_;
	long numImages_;
	long imageCounter_;
	double intervalMs_;
	MM::MMTime startTime_;
	MM::MMTime actualDuration_;
	MM::MMTime lastFrameTime_;
	MMThreadLock stopLock_;
	MMThreadLock suspendLock_;
	NikonKsCam* camera_;
};


#endif //_NIKONKS_H_

