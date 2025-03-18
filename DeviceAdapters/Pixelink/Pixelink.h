

#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/ImgBuffer.h"
#include "../../MMDevice/DeviceThreads.h"
#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <PixeLINKAPI.h>



//////////////////////////////////////////////////////////////////////////////
// Error codes
//

#define max(a,b)            (((a) > (b)) ? (a) : (b))
#define min(a,b)            (((a) < (b)) ? (a) : (b))


//////////////////////////////////////////////////////////////////////////////

class MySequenceThread;



struct PixelAddressingPair
{
	std::string binName;
	int		x;
	int     y;
};


struct PixelAddressing
{
	int		Mode;
	int		x;
	int     y;
	bool	supportAsymmetry; // true indicates the camera will accept dirffert x and y values
};

float GetPixelSize(U32 pixelFormat);
U32 DetermineRawImageSize(HANDLE hCamera);

class Pixelink : public CLegacyCameraBase<Pixelink>
{
public:
	Pixelink(const char* deviceName);
	~Pixelink();

	//////////////////////////////////////////////////////////////
	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;

	int SnapImage();
	const unsigned char* GetImageBuffer();
	unsigned int GetNumberOfComponents()  const { return nComponents_; };
	unsigned int GetImageWidth() const;
	unsigned int GetImageHeight() const;
	unsigned int GetImageBytesPerPixel() const;
	unsigned int GetBitDepth() const;
	long     GetImageBufferSize() const;
	double   GetExposure() const;
	void     SetExposure(double exp);
	void SetCameraDevice(const char* cameraLabel);
	double GetCameraDevice() const;
	int      SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int      GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int      ClearROI();
	int      PrepareSequenceAcqusition() { return DEVICE_OK; };
	int      StartSequenceAcquisition(double interval);
	int      StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int      StopSequenceAcquisition();
	bool     IsCapturing();
	int      GetBinning() const;
	int      SetBinning(int binSize);
	int      IsExposureSequenceable(bool& seq) const { seq = false; return DEVICE_OK; }
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPaMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	HANDLE m_handle;
	float ReturnFeature(HANDLE hCamera, U32 featureId, int multiplier = 1);
	std::vector<U8> image_;
	PXL_RETURN_CODE GetNextFrame(HANDLE hCamera, U32 frameBufferSize, void* pFrameBuffer, FRAME_DESC* pFrameDesc, U32 maximumNumberOfTries) const;
	ImgBuffer       img_;
	FRAME_DESC frameDesc;
	PCAMERA_FEATURES				m_pFeatures;
	std::vector<CAMERA_FEATURE*>	m_features;
	int GetPixelFormatSize(int pixelFormat) const;
	void Pixelink::GenerateEmptyImage(ImgBuffer& img);
	bool Pixelink::SupportedPixelType(int i);
	unsigned int GetPixelFormatByteSize(int pixelFormat) const;
	MMThreadLock imgPixelsLock_;
	void SetFeatureRoi(float left, float top, float width, float height, bool cameraStreaming, bool isClearingRoi);
	int SetGain(float gain);
	float GetGain();
	int InsertImage();
    static U32 __stdcall CallbackFrameFromCamera(HANDLE hCamera,LPVOID pData,U32 dataFormat,FRAME_DESC const *	pFrameDesc,LPVOID userData);
	int RunSequenceOnThread(MM::MMTime startTime);
	void OnThreadExiting() throw();
	static bool capturedOneFrame;
	bool IsColourCamera();
	int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
	std::string PixelTypeAsString(int pixelFormat, bool asCodeConst) const;
	int SetPixelFormat(float pf);
	float GetPixelFormat();
	unsigned char* pPixel;
	void LoadSupportedPaValues();
	bool                            m_supportsAsymmetricPa;
	std::vector<int>				m_supportedXDecimations;
	std::vector<int>				m_supportedYDecimations;
	std::vector<int>				m_supportedDecimationModes;
	std::vector<int>				m_supportedPixelFormats;
	std::vector<PixelAddressingPair>				m_supportedDecimationModePairs;
	void LoadCameraFeatures(void);
	void ClearFeatures(void);
	void UpdateFrameBufferSize(int fHorizontal, int fVertical);
	bool isStreaming;
	PixelAddressing GetPixelAddressing(U32* pFlags = NULL);
	PXL_RETURN_CODE SetPixelAddressing(const PixelAddressing pixelAddressing);
	CAMERA_FEATURE* GetFeaturePtr(const U32 featureId);
	void LoadSupportedPixelFormats();
	void SetPixelType(int pixelType);
	int StringAsPixelType(std::string pixelFormat) const;


private:

	unsigned char* RGB24ToRGBA(const unsigned char* img, int width, int height);
	unsigned char* RGB48ToRGBA(const unsigned char* img, int width, int height);
	unsigned char* Mono16ToBuffer(const unsigned char* img, int width, int height);
	unsigned char* Mono12ToBuffer(const unsigned char* img, int width, int height);

	unsigned int nComponents_;
	bool initialized_;
	std::string deviceName_;
	MM::MMTime sequenceStartTime_;
	MM::MMTime sequenceStartTimeStamp_;
	long imageCounter_;
	bool stopOnOverflow_;
	long desiredNumImages_;
	bool isCapturing_;
	std::map<long, std::string> bin2Mode_;
	std::map<const std::string, long> mode2Bin_;
	std::vector<unsigned short> availableTriggerModes_;
	bool f7InUse_;
	double exposureTimeMs_;
	unsigned short triggerMode_;
	unsigned short snapTriggerMode_;
	unsigned long externalTriggerGrabTimeout_;
	unsigned short bytesPerPixel_;
	MMThreadLock imgBuffLock_;
	unsigned char* imgBuf_;
	const unsigned long bufSize_;
	unsigned int bitDepth_;
	void CreateGainSlider();
	int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);


	friend class MySequenceThread;
	MySequenceThread * thd_;

	int tempCounter;

	std::string PixelFormatAsString(int format);

	int StringasPixelFormat(char* pixelFormat);
};


class MySequenceThread : public MMDeviceThreadBase
{
	friend class Pixelink;
	enum { default_numImages = 1, default_intervalMS = 100 };
public:
	MySequenceThread(Pixelink* pCam);
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
	Pixelink* camera_;
	MM::MMTime startTime_;
	MM::MMTime actualDuration_;
	MM::MMTime lastFrameTime_;
	MMThreadLock stopLock_;
	MMThreadLock suspendLock_;
};