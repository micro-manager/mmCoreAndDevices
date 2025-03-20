// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the MTUSBDLL_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// MTUSBDLL_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.

#define GRAB_FRAME_FOREVER	0x8888

#pragma pack(1)

typedef struct {
    int CameraID;
    int WorkMode;	 // 0 - NORMAL mode, 1 - TRIGGER mode
    int SensorClock; // It's reserved for SB camera.
    int Row;	// It's ColSize, in 1408x1088, it's 1088
    int Column;	// It's RowSize, in 1408x1088, it's 1408
    int Bin;	// 0, 1, for no-decimation  and 1:2 1:4 decimation
    int BinMode;// 0 - Skip, 1 - Bin
    int CameraBit; // 8 or 16.
    int XStart;
    int YStart;
    int ExposureTime; // in 50us unit, e.g. 100 means 5000us(5ms)
    int RedGain;
    int GreenGain;
    int BlueGain;
    int TimeStamp; // Cast it to "unsigned int" when using it
    int SensorMode; //  Reserved
    int TriggerOccurred; // Reserved
    int TriggerEventCount; // Reserved.
    int FrameSequenceNo; // Reserved.
    int IsFrameBad; // Is the current frame a bad one.

    int FrameProcessType; // Bit0: 0 - RAW, 1 - BMP
    int FilterAcceptForFile; // Reserved
} TProcessedDataProperty;

#pragma pack()

typedef void (* DeviceFaultCallBack)( int DeviceID, int DeviceType );
typedef void (* FrameDataCallBack)( TProcessedDataProperty* Attributes, unsigned char *BytePtr );

// Import functions:
typedef int (WINAPI * SSBufferUSB_InitDevicePtr)( void );
typedef int (WINAPI * SSBufferUSB_UnInitDevicePtr)( void );
typedef int (WINAPI * SSBufferUSB_GetModuleNoSerialNoPtr)( int DeviceID, char *ModuleNo, char *SerialNo);
typedef int (WINAPI * SSBufferUSB_AddDeviceToWorkingSetPtr)( int DeviceID );
typedef int (WINAPI * SSBufferUSB_RemoveDeviceFromWorkingSetPtr)( int DeviceID );
typedef int (WINAPI * SSBufferUSB_ActiveDeviceInWorkingSetPtr)( int DeviceID, int Active );
typedef int (WINAPI * SSBufferUSB_StartCameraEngineExPtr)( int ParentHandle, int CameraBitOption, int ProcessThreads, int IsCallBackInThread );
typedef int (WINAPI * SSBufferUSB_StartCameraEnginePtr)( int ParentHandle, int CameraBitOption );
typedef int (WINAPI * SSBufferUSB_StopCameraEnginePtr)( void );
typedef int (WINAPI * SSBufferUSB_SetUSBConnectMonitorPtr)( int DeviceID, int MonitorOn );
typedef int (WINAPI * SSBufferUSB_SetUSB30TransferSizePtr)( int TransferSizeLevel );
typedef int (WINAPI * SSBufferUSB_GetCameraFirmwareVersionPtr)( int DeviceID, int FirmwareType );
typedef int (WINAPI * SSBufferUSB_StartFrameGrabExPtr)( int DeviceID, int TotalFrames );
typedef int (WINAPI * SSBufferUSB_StopFrameGrabExPtr)( int DeviceID );
typedef int (WINAPI * SSBufferUSB_StartFrameGrabPtr)( int TotalFrames );
typedef int (WINAPI * SSBufferUSB_StopFrameGrabPtr)( void );
typedef int (WINAPI * SSBufferUSB_ShowFactoryControlPanelPtr)( int DeviceID, char *passWord );
typedef int (WINAPI * SSBufferUSB_HideFactoryControlPanelPtr)( void );
typedef int (WINAPI * SSBufferUSB_SetBayerFilterTypePtr)( int DeviceID, int FilterType );
typedef int (WINAPI * SSBufferUSB_SetCameraWorkModePtr)( int DeviceID, int WorkMode );
typedef int (WINAPI * SSBufferUSB_SetCustomizedResolutionExPtr)( int deviceID, int RowSize, int ColSize, int Bin, int BinMode, int BufferCnt, int BufferOption );
typedef int (WINAPI * SSBufferUSB_SetCustomizedResolutionPtr)( int deviceID, int RowSize, int ColSize, int Bin, int BufferCnt );
typedef int (WINAPI * SSBufferUSB_SetExposureTimePtr)( int DeviceID, int exposureTime );
typedef int (WINAPI * SSBufferUSB_SetFrameTimePtr)( int DeviceID, int frameTime );
typedef int (WINAPI * SSBufferUSB_SetXYStartPtr)( int DeviceID, int XStart, int YStart );
typedef int (WINAPI * SSBufferUSB_SetGainsPtr)( int DeviceID, int RedGain, int GreenGain, int BlueGain );
typedef int (WINAPI * SSBufferUSB_SetGainRatiosPtr)( int DeviceID, int RedGainRatio, int BlueGainRatio);
typedef int (WINAPI * SSBufferUSB_SetGammaPtr)( int DeviceID, int Gamma, int Contrast, int Bright, int Sharp );
typedef int (WINAPI * SSBufferUSB_SetBWModePtr)( int DeviceID, int BWMode, int H_Mirror, int V_Flip );
typedef int (WINAPI * SSBufferUSB_SetMinimumFrameDelayPtr)( int IsMinimumFrameDelay ); 
typedef int (WINAPI * SSBufferUSB_SoftTriggerPtr)( int DeviceID );
typedef int (WINAPI * SSBufferUSB_SetSensorBlankingsPtr)( int DeviceID, int HBlanking, int VBlanking );
typedef int (WINAPI * SSBufferUSB_InstallFrameCallbackPtr)( int FrameType, FrameDataCallBack FrameHooker );
typedef int (WINAPI * SSBufferUSB_InstallUSBDeviceCallbackPtr)( DeviceFaultCallBack USBDeviceHooker );
typedef int (WINAPI * SSBufferUSB_InstallFrameHookerPtr)( int FrameType, FrameDataCallBack FrameHooker );
typedef int (WINAPI * SSBufferUSB_InstallUSBDeviceHookerPtr)( DeviceFaultCallBack USBDeviceHooker );
typedef unsigned char * (WINAPI * SSBufferUSB_GetCurrentFramePtr)( int FrameType, int DeviceID, unsigned char* &FramePtr );
typedef unsigned short * (WINAPI * SSBufferUSB_GetCurrentFrame16bitPtr)( int FrameType, int DeviceID, unsigned short* &FramePtr );
typedef unsigned long * (WINAPI * SSBufferUSB_GetCurrentFrameParaPtr)( int DeviceID, unsigned long* &FrameParaPtr );
typedef int (WINAPI * SSBufferUSB_GetDevicesErrorStatePtr)();
typedef int (WINAPI * SSBufferUSB_IsUSBSuperSpeedPtr)( int DeviceID );
typedef int (WINAPI * SSBufferUSB_SetGPIOConfigPtr)( int DeviceID, unsigned char ConfigByte );
typedef int (WINAPI * SSBufferUSB_SetGPIOOutPtr)( int DeviceID, unsigned char OutputByte );
typedef int (WINAPI * SSBufferUSB_SetGPIOInOutPtr)( int DeviceID, unsigned char OutputByte, unsigned char *InputBytePtr );


