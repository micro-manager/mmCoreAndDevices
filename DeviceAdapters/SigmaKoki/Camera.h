///////////////////////////////////////////////////////////////////////////////
// FILE:          Camera2.h
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#pragma once
#pragma region Prehead_Inclus
#include "SigmaBase.h"
#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
using namespace std;
extern const char* g_CameraDeviceName;
#pragma endregion Prehead_Inclus

#pragma region Camera_Err_MSG
#define ERR_CAMERA_OPEN_FAILED								10401
#define ERR_CAMERA_GET_PREVIEWDATASIZE_FAILED				10402
#define ERR_CAMERA_SET_PIXELFORMAT_FAILED					10403
#define ERR_CAMERA_SET_TARGET_BRIGHTNESS_FAILED				10404
#define ERR_CAMERA_GET_SHUTTER_GAIN_CONTROL_RANGE_FAILED	10405
#define ERR_CAMERA_SET_ALCMODE_FAILED						10406
#define ERR_CAMERA_GET_ALCMODE_FAILED						10407
#define ERR_CAMERA_SET_SHUTTER_GAIN_FAILED					10408
#define ERR_CAMERA_GET_SHUTTER_GAIN_FAILED					10409
#define ERR_CAMERA_GET_EXPOSURE_TIME_FAILED					10410
#define ERR_CAMERA_SET_EXPOSURE_TIME_FAILED					10411
#define ERR_CAMERA_GET_CLOCK_FAILED							10412
#define ERR_CAMERA_SET_CLOCK_FAILED							10413
#define ERR_CAMERA_GET_FPS_FAILED							10414
#define ERR_CAMERA_START_TRANSFER_FAILED					10415
#define ERR_CAMERA_STOP_TRANSFER_FAILED						10416
#define ERR_CAMERA_SNAPSHOT_FAILED							10417
#define ERR_CAMERA_LIVE_STOP_UNKNOWN						10418
#define ERR_CAMERA_GET_PRODUCT_NAME_FAILED					10419
#define ERR_CAMERA_SET_BINNING_SCAN_MODE_FAILED				10420
#define ERR_CAMERA_GET_BINNING_SCAN_MODE_FAILED				10421
#define ERR_CAMERA_SET_WB_MODE_FAILED						10422
#define ERR_CAMERA_GET_WB_MODE_FAILED						10423
#define ERR_CAMERA_SET_WB_GAIN_FAILED						10424
#define ERR_CAMERA_GET_WB_GAIN_FAILED						10425
#define ERR_CAMERA_GET_COLOR_ARRAY_FAILED					10426
#define ERR_CAMERA_SET_ROI_FAILED							10427
#define ERR_CAMERA_GET_ROI_COUNT_FAILED						10428
#define ERR_CAMERA_ALCMODE_UNAVAILABLE_FUNCTION				10429
#define ERR_CAMERA_WBMODE_UNAVAILABLE_FUNCTION				10430
#define ERR_CAMERA_SCAN_MODE_FAILED_SETTING  				10431
#define ERR_CAMERA_SCAN_MODE_PROHIBTED						10432

#pragma endregion Camera_Err_MSG

class Camera : public CLegacyCameraBase<Camera>, public SigmaBase
{
#pragma region Constructor_Des 
public:
	Camera();
	~Camera();
#pragma endregion Constructor_Des

#pragma region Device_Api
	void GetName(char* pszName) const;
	int Initialize();
	int Shutdown();
#pragma endregion Device_Api

#pragma region Camera_Api
	int SnapImage();
	const unsigned char* GetImageBuffer();
	long GetImageBufferSize() const;
	unsigned GetImageWidth() const;
	unsigned GetImageHeight() const;
	unsigned GetImageBytesPerPixel() const;
	unsigned GetBitDepth() const;
	int GetBinning() const;
	int SetBinning(int binSize);
	double GetExposure() const;
	void SetExposure(double exp_ms);
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int ClearROI();
	int PrepareSequenceAcqusition() { return DEVICE_OK; };
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StartSequenceAcquisition(double interval_ms);
	int StopSequenceAcquisition();
	int InsertImage();
	int ThreadRun();
	bool IsCapturing();
	void OnThreadExiting() throw();
	int IsExposureSequenceable(bool& isSequenceable) const;
	unsigned  GetNumberOfComponents() const { return nComponents_; };
#pragma endregion Camera_Api
	
#pragma region Action_interface

	int OnALCMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGainMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnShutterGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoGainMax(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoGainMin(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnClockSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWhiteBalanceMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWBGainR(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWBGainGr(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWBGainGb(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWBGainB(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnReadoutTime(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMirrorRotation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnImageH(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnImageV(MM::PropertyBase* pProp, MM::ActionType eAct);
#pragma endregion Action_interface 

#pragma region Declaration_Variable 
private:
	int ResizeImageBuffer();
	HANDLE handle_;
	char name_;
	bool isMonochrome_;
	bool enabledROI_;
	std::string alcMode_, scanModeType_, productName_;
	DWORD dwSize_;
	
	DWORD dwLinePitch_;
	unsigned int dwPreviewPixelFormat_;
	WORD gain_, scanMode_;
	double exposureMsec_;
	DWORD exposureClock_;
	WORD autoGainMax_, analogGainMax_, digitalGainMax_, digitalGainMin_;
	WORD autoGainMin_;
	double exposureMaxMsec_;
	double exposureMinMsec_;
	DWORD exposureMaxClock_;
	std::string clockMode_, mirrorMode_, gainMode_;   // mirror command + Gain Mode added.(8/3/22)
	DWORD clockFreq_;
	FLOAT fps_;
	std::string wbMode_;
	WORD wbGainR_;
	WORD wbGainGr_;
	WORD wbGainGb_;
	WORD wbGainB_;
	unsigned int delayMsec_;
	ImgBuffer img_;
	bool busy_;
	bool stopOnOverFlow_;
	bool initialized_;
	double readoutUs_;
	MM::MMTime readoutStartTime_;
	MM::MMTime sequenceStartTime_;
	int bitDepth_;
	unsigned roiX_;
	unsigned roiY_;
	bool isSequenceable_;
	long imageCounter_;
	long binFac_;
	DWORD imageSizeH_;
	DWORD imageSizeV_;
	bool stopOnOverflow_;
	MMThreadLock imgPixelsLock_;
	friend class MySequenceThread;
	int nComponents_;
	MySequenceThread* thd_;
#pragma endregion Declaration_Variable 

};

class MySequenceThread : public MMDeviceThreadBase
{
#pragma region Live_Api 
	friend class Camera;
	enum { default_numImages = 1, default_intervalMS = 100 };
public:
	MySequenceThread(Camera* pCam);
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
#pragma endregion Live_Api

#pragma region Live_Declaration
private:
	int svc(void) throw();
	double intervalMs_;
	long numImages_;
	long imageCounter_;
	bool stop_;
	bool suspend_;
	Camera* camera_;
	MM::MMTime startTime_;
	MM::MMTime actualDuration_;
	MM::MMTime lastFrameTime_;
	MMThreadLock stopLock_;
	MMThreadLock suspendLock_;
#pragma endregion Live_Declaration
};
