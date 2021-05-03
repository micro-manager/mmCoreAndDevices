///////////////////////////////////////////////////////////////////////////////
// FILE:          Basler_mE5AQ8-CXP6D_CameraDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device Adapter for Basler microEnable5 AQ8-CXP6D
//                framegrabber board
//                (formerly: Silicon Software mE5AQ8-CXP6D)
// COPYRIGHT:
//                Copyright 2021 BST
//
// VERSION:		  1.0.0.1 
//
// LICENSE:
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
//
// HISTORY:
//           BST : 04/01/2021 Initial Release
//


#ifndef _Basler_mE5AQ8CXP6D_H_
#define _Basler_mE5AQ8CXP6D_H_
#pragma once
#include "../../../MMDevice/MMDevice.h"
#include "../../../MMDevice/DeviceBase.h"
#include <string>
#include <map>

#include <fgrab_struct.h>
#include <fgrab_prototyp.h>

//#define GENICAM_USER_ALWAYS_LINK_RELEASE 
#include <siso_genicam.h>

#include "SiSoGenICamCamera.h"

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_FG_INITLIBRARIES_FAILED	101
#define ERR_FG_GETSYSTEMINFORMATION_FAILED 102
#define ERR_NO_CAMERA_FOUND 103
#define ERR_SERIAL_NUMBER_REQUIRED 104
#define ERR_NODE_NOT_READABLE 105
#define ERR_NODE_NOT_WRITABLE 106
#define ERR_CONFIGFILE_NOT_FOUND 107
#define ERR_INITCONFIG_FAILED 108
#define ERR_XML_FROM_CAMERA_FAILED 109
#define ERR_FG_GET_PARAMETER 110






typedef struct S_DEVICELISTENTRY
{
	int BoardType;
	int BoardIndex;
	std::string BoardSerialNumber;
	int PortIndex;
	std::string CameraSerialNumber;
	std::string CameraModelName;
} DeviceListEntry_t;
typedef std::vector<DeviceListEntry_t> DeviceList_t;

// Basler_mE5AQ8-CXP6D_CameraDevice.cpp
extern const char* g_DeviceName_mE5AQ8CXP6D_CameraDevice;
class CmE5AQ8CXP6D_CameraDevice;
struct fg_apc_data {
	Fg_Struct *fg;
	unsigned int port;
	dma_mem *mem;
	int displayid;
	CmE5AQ8CXP6D_CameraDevice* pThis;
};


class CSiSoGenICamCamera;
// CCameraBase<U>
// --------------
class CmE5AQ8CXP6D_CameraDevice : public CCameraBase<CmE5AQ8CXP6D_CameraDevice>  
{
public:
	CmE5AQ8CXP6D_CameraDevice();
	~CmE5AQ8CXP6D_CameraDevice();

	// MM::Device
	// ----------
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy();
	void AddToLog(std::string msg);

	// CCameraBase<U>
	// --------------
	int SnapImage();
	unsigned char const* GetImageBuffer();
	unsigned int GetImageWidth() const;
	unsigned int GetImageHeight() const;
	unsigned int GetImageBytesPerPixel() const;

	// MM::Camera
	// --------
	long GetImageBufferSize() const;
	unsigned int GetBitDepth() const;
	int GetBinning() const;
	int SetBinning(int binSize);
	void SetExposure(double exp);
	double GetExposure() const;
	int SetROI(unsigned int x, unsigned int y, unsigned int xSize, unsigned int ySize); 
	int GetROI(unsigned int& x, unsigned int& y, unsigned int& xSize, unsigned int& ySize);
	int ClearROI();
	int IsExposureSequenceable(bool& seq) const {seq = false; return DEVICE_OK;}

	// Property handlers
	//------------------
	int OnBinning(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int On_FG_Width(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_Height(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_XOffset(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_YOffset(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_Format(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_PixelDepth(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_BitAlignment(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_FG_PixelFormat(MM::PropertyBase* pProp, MM::ActionType pAct);

	int On_CAM_PixelFormat(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_Width(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_Height(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_OffsetX(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_OffsetY(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_ExposureMode(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_ExposureTime(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_ExposureTimeRaw(MM::PropertyBase* pProp, MM::ActionType pAct);

	int On_CAM_BinningMode(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_ReverseX(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_ReverseY(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_TestImageSelector(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_TestImageVideoLevel(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_CrosshairOverlay(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_AcquisitionFrameRate(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_AcquisitionMaxFrameRate(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_TriggerSource(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_TriggerActivation(MM::PropertyBase* pProp, MM::ActionType pAct);
	int On_CAM_AcquisitionMode(MM::PropertyBase* pProp, MM::ActionType pAct);
	






	/**
	* Starts continuous acquisition.
	*/
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StartSequenceAcquisition(double interval_ms);
	int StopSequenceAcquisition();
	int PrepareSequenceAcqusition();

	/**
	* Flag to indicate whether Sequence Acquisition is currently running.
	* Return true when Sequence acquisition is active, false otherwise
	*/
	bool IsCapturing();

	std::string FG_GetOutputFormat();

private:

//	static MMThreadLock lock_;
//	static int ApcFunc(frameindex_t picNr, struct fg_apc_data *data);

	bool siso_libraries_initialized_;
	bool framegrabber_initialized_;
	bool sgc_board_initialized_;
	bool sgc_camera_connected_;
	bool sgc_link_connected_;
	bool mm_device_initialized_;
	int camera_enumeration_error_;
	DeviceList_t camera_device_list_;
	DeviceListEntry_t selected_camera_;
	Fg_Struct* framegrabber_;
	SgcBoardHandle *sgc_board_handle_;
	SgcCameraHandle* sgc_camera_handle_;
	CSiSoGenICamCamera *genicam_camera_;
	void *snap_buffer_;
	long snap_buffer_size_;
	bool is_sequence_acquisition_;
	struct FgApcControl apc_ctrl_;
	struct fg_apc_data apcdata_;
	dma_mem *dma_mem_;




	DeviceList_t EnumerateCameras();
	void ResizeSnapBuffer();

	int FG_GetEnumNameFromEnumValue(int ParameterId, int Value, std::string& EnumName);
	int FG_GetEnumValueFromEnumName(int ParameterId, std::string& EnumName, int *Value);
	int FG_GetDisplayNameFromEnumValue(int ParameterId, int Value, std::string& EnumName);
	int FG_GetEnumValueFromDisplayName(int ParameterId, std::string& DisplayName, int *Value);

#if 0
   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();
   static MMThreadLock& GetLock() {return lock_;}
#endif
};



#endif //_Basler_mE5AQ8CXP6D_H_