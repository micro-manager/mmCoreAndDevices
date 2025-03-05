#pragma once

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ASICamera2.h"
#include "EFW_filter.h"
#include "ModuleInterface.h"
#include "comdef.h"

class SequenceThread;

class CMyASICam:public CLegacyCameraBase<CMyASICam> 
{
public:
	CMyASICam(void);
	~CMyASICam(void);
	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* name) const;      

	// MMCamera API
	// ------------
	int SnapImage();
	const unsigned char* GetImageBuffer();
//	const unsigned int* GetImageBufferAsRGB32();
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
	bool IsCapturing();
	int GetBinning() const;
	int SetBinning(int binSize);

	int IsExposureSequenceable(bool& seq) const {seq = false; return DEVICE_OK;}
	int PrepareSequenceAcqusition();
	int StartSequenceAcquisition(double interval);
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StopSequenceAcquisition();
	unsigned  GetNumberOfComponents() const { return iComponents;};
	int GetComponentName(unsigned component, char* name);
	// action interface
	// ----------------
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnUSBTraffic(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSelectCamIndex(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCoolerOn(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHeater(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTargetTemp(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCoolerPowerPerc(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWB_R(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWB_B(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoWB(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnUSB_Auto(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoExp(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFlip(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHighSpeedMod(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHardwareBin(MM::PropertyBase* pProp, MM::ActionType eAct);

private:


	static const int MAX_BIT_DEPTH = 16;


	long lExpMs;

	enum CamStatus {
		closed = 0,
		opened,
		capturing,
		snaping
	};
	CamStatus Status;
	

	char FlipArr[4][8];
	char ConnectedCamName[32][32];
	bool initialized_;
	int iConnectedCamNum;

	//variable of a camera
	unsigned char *uc_pImg;
	unsigned char *pRGB32, *pRGB64;
	unsigned long iBufSize;


	int iPixBytes;//每个像素字节数	  
	int iComponents;
	int   iROIWidth, iROIHeight, iBin;//sensor的坐标尺寸信息
	int   iSetWid, iSetHei, iSetBin, iSetX, iSetY; //要设置的坐标尺寸信息

	ASI_IMG_TYPE ImgType;
	friend class SequenceThread;
	SequenceThread* thd_;
	ASI_CONTROL_CAPS* pControlCaps;
	ASI_CAMERA_INFO ASICameraInfo;
	int iCtrlNum;
	ASI_FLIP_STATUS ImgFlip;
	int ImgStartX, ImgStartY, ImgBin, ImgWid, ImgHei;//所显示图像的坐标尺寸信息


//	int iCamIndex;
	char sz_ModelIndex[64];
	bool b12RAW, bRGB48;
	void DeleteImgBuf();
	int RunSequenceOnThread(MM::MMTime startTime);
	int InsertImage();
	long imageCounter_;
	void MallocControlCaps(int iCamindex);
	void DeletepControlCaps(int iCamindex);
	bool isImgTypeSupported(ASI_IMG_TYPE ImgType);

	ASI_CONTROL_CAPS* GetOneCtrlCap(int CtrlID);
	void ConvRGB2RGBA32();
	void ConvRGB2RGBA64();
	void Conv16RAWTo12RAW();
	void RefreshImgType();
};


class SequenceThread : public MMDeviceThreadBase
{
public:
	SequenceThread(CMyASICam* pCam);
	~SequenceThread();
	void Stop();
	void Start(long numImages, double intervalMs);
	bool IsStopped();
	double GetIntervalMs(){return intervalMs_;}                               
	void SetLength(long images) {numImages_ = images;}                        
	long GetLength() const {return numImages_;}
	long GetImageCounter(){return imageCounter_;} 

private:                                                                     
	int svc(void) throw();
	CMyASICam* camera_;                                                     
	bool stop_;                                                               
	long numImages_;                                                          
	long imageCounter_;                                                       
	double intervalMs_;                                                       
}; 

class CMyEFW : public CStateDeviceBase<CMyEFW>
{
public:
	CMyEFW();
	~CMyEFW();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();
	unsigned long GetNumberOfPositions() const;

	// action interface
	// ----------------
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
//	int OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSelectEFWIndex(MM::PropertyBase* pProp, MM::ActionType eAct);
//	int GetPosition(long& pos) const;
//	int SetPosition(long pos);
//	int GetPosition(long& pos);
private:
//	long numPos_;
	EFW_INFO EFWInfo;
	int iConnectedEFWNum;
	long lLastPos;
	bool initialized_;
	MM::MMTime changedTime_;
	char ConnectedEFWName[32][32];
	char sz_ModelIndex[64];
	bool bPosWait;
//	long position_;
};
//2.0.0.0->20170113:增加anti-dew和EFW