///////////////////////////////////////////////////////////////////////////////
// FILE:          Basler_mE5AQ8-CXP6D_CameraDevice.cpp
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



#include "Basler_mE5AQ8-CXP6D_CameraDevice.h"
#include "../../../MMDevice/ModuleInterface.h"
#include <sstream>
#include <cstdio>


#include <fgrab_prototyp.h>
#include <fgrab_struct.h>
#include <siso_genicam.h>

#include <GenICam.h>

#include "SiSoGenICamCamera.h"

using namespace std;
using namespace GenApi;

//#ifdef WIN32
//   #define WIN32_LEAN_AND_MEAN
//   #include <windows.h>
//#endif
//#include "FixSnprintf.h"


// C-style AsynchronousProcedureCallback for retrieving images in SequenceAcquisitionMode
extern "C" int ApcFunc(frameindex_t picNr, struct fg_apc_data *data)
{
	// Important:  meta data about the image are generated here:
	Metadata md;
	md.put("Camera", "");
	md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)data->pThis->GetImageWidth()));
	md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)data->pThis->GetImageHeight()));
	md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString((long)picNr));
	md.put(MM::g_Keyword_Meatdata_Exposure, data->pThis->GetExposure());

    void *ptr_image = Fg_getImagePtrEx(data->fg, picNr, data->port, data->mem);
    size_t len;
    int siso_result = Fg_getParameterEx(data->fg, FG_TRANSFER_LEN, &len, data->port, data->mem, picNr);
	int fg_format;
	siso_result = Fg_getParameter(data->fg, FG_FORMAT, &fg_format, data->port);
	switch (fg_format)
	{
	case FG_GRAY:
	case FG_GRAY10:
	case FG_GRAY12:
	case FG_GRAY14:
	case FG_GRAY16:
	case FG_GRAY32:
		{
			//copy to intermediate buffer
			int ret = data->pThis->GetCoreCallback()->InsertImage(data->pThis, (const unsigned char*)ptr_image,
				(unsigned)data->pThis->GetImageWidth(), (unsigned)data->pThis->GetImageHeight(),
				(unsigned)data->pThis->GetImageBytesPerPixel(), 1, md.Serialize().c_str(), FALSE);
			if (ret == DEVICE_BUFFER_OVERFLOW)
			{
				//if circular buffer overflows, just clear it and keep putting stuff in so live mode can continue
				data->pThis->GetCoreCallback()->ClearImageBuffer(data->pThis);
			}
		}
		break;
	default:
		{
			char s[255];
			sprintf(s, "ApcFunc(): fg_format( %d ) not yet implemented.\n", fg_format);
			data->pThis->AddToLog(std::string(s));
			return DEVICE_NOT_YET_IMPLEMENTED;
		}
	}
	return DEVICE_OK;
}


const char* g_DeviceName_mE5AQ8CXP6D_CameraDevice = "mE5AQ8CXP6D";//_CameraDevice";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName_mE5AQ8CXP6D_CameraDevice, MM::CameraDevice, "CmE5AQ8CXP6D_CameraDevice");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceName_mE5AQ8CXP6D_CameraDevice) == 0)
   {
      return new CmE5AQ8CXP6D_CameraDevice;
   }
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// CmE5AQ8CXP6D implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
CmE5AQ8CXP6D_CameraDevice::CmE5AQ8CXP6D_CameraDevice() :
	siso_libraries_initialized_(false),
	framegrabber_initialized_(false),
	framegrabber_(NULL),
	sgc_board_initialized_(false),
	sgc_board_handle_(NULL),
	sgc_camera_handle_(NULL),
	sgc_camera_connected_(false),
	sgc_link_connected_(false),
	mm_device_initialized_ (false),
	camera_enumeration_error_(DEVICE_OK),
	snap_buffer_(NULL),
	snap_buffer_size_(0),
	genicam_camera_(NULL),
	is_sequence_acquisition_(false)
{
	InitializeDefaultErrorMessages();

	//////////////////////////////////////////////////////////////////////////////
	// Error codes defined in header file
	//
	// Set default error texts
	//
	//#define ERR_FG_INITLIBRARIES_FAILED	101
	//#define ERR_FG_GETSYSTEMINFORMATION_FAILED 102
	//#define ERR_NO_CAMERA_FOUND 103
	//#define ERR_SERIAL_NUMBER_REQUIRED 104
	//#define ERR_NODE_NOT_READABLE 105
	//#define ERR_NODE_NOT_WRITABLE 106
	//#define ERR_CONFIGFILE_NOT_FOUND 107
	//#define ERR_INITCONFIG_FAILED 108
	//#define ERR_XML_FROM_CAMERA_FAILED 109
	//#define ERR_FG_GET_PARAMETER 110
	SetErrorText(ERR_FG_INITLIBRARIES_FAILED, "Fg_InitLibraries() failed.");
	SetErrorText(ERR_FG_GETSYSTEMINFORMATION_FAILED, "Fg_getSystemInformation() failed.");
	SetErrorText(ERR_NO_CAMERA_FOUND, "No Camera found!");
	SetErrorText(ERR_SERIAL_NUMBER_REQUIRED, "Serial number is required.");
	SetErrorText(ERR_NODE_NOT_READABLE, "Property is not readable.");
	SetErrorText(ERR_NODE_NOT_WRITABLE, "Property is not writable.");
	SetErrorText(ERR_CONFIGFILE_NOT_FOUND, "Configfile not found.");

	// PreInitProperty: Select Camera by SerialNumber
	CreateProperty("Camera Selector", "No Camera found!", MM::String, false, 0, true);

	camera_device_list_.clear();
	camera_device_list_ = EnumerateCameras();
	bool first = true;
	for (DeviceList_t::iterator it = camera_device_list_.begin(); it != camera_device_list_.end(); it++)
	{
		AddAllowedValue("Camera Selector", (*it).CameraSerialNumber.c_str());
		if (first)
		{
			SetProperty("Camera Selector", (*it).CameraSerialNumber.c_str());
			first = false;
		}
	}
}

CmE5AQ8CXP6D_CameraDevice::~CmE5AQ8CXP6D_CameraDevice()
{
	Shutdown();
}

void CmE5AQ8CXP6D_CameraDevice::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceName_mE5AQ8CXP6D_CameraDevice);
}

bool CmE5AQ8CXP6D_CameraDevice::Busy()
{
   return false;
}

int CmE5AQ8CXP6D_CameraDevice::FG_GetEnumNameFromEnumValue(int ParameterId, int Value, std::string& EnumName)
{
	int buf_len = 0;
	Fg_getParameterPropertyEx(framegrabber_, ParameterId, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, NULL, &buf_len);
	char *buf = (char*)malloc(buf_len);
	Fg_getParameterPropertyEx(framegrabber_, ParameterId, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, buf, &buf_len);
	FgPropertyEnumValues* e = (FgPropertyEnumValues*)buf;
	std::vector<std::string> enums;
	while (e->value != -1)
	{
		if (e->value == (int32_t)Value)
		{
			EnumName.assign(e->name);
			free(buf);
			return FG_OK;
		}
		e = FG_PROP_GET_NEXT_ENUM_VALUE(e);
	}
	free(buf);
	return FG_ERROR;
}

int CmE5AQ8CXP6D_CameraDevice::FG_GetEnumValueFromEnumName(int ParameterId, std::string& EnumName, int *Value)
{
	int buf_len = 0;
	Fg_getParameterPropertyEx(framegrabber_, ParameterId, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, NULL, &buf_len);
	char *buf = (char*)malloc(buf_len);
	Fg_getParameterPropertyEx(framegrabber_, ParameterId, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, buf, &buf_len);
	FgPropertyEnumValues* e = (FgPropertyEnumValues*)buf;
	std::vector<std::string> enums;
	while (e->value != -1)
	{
		if (strcmp(EnumName.c_str(), e->name) == 0)
		{
			*Value = (int)e->value;
			free(buf);
			return FG_OK;
		}
		e = FG_PROP_GET_NEXT_ENUM_VALUE(e);
	}
	free(buf);
	return FG_ERROR;
}

int CmE5AQ8CXP6D_CameraDevice::FG_GetDisplayNameFromEnumValue(int ParameterId, int Value, std::string& DisplayName)
{
	switch(ParameterId)
	{
	case FG_FORMAT:
		switch(Value)
		{
		case FG_GRAY:
			DisplayName = "Gray 8bit";
			return FG_OK;
		default:
			return FG_GetEnumNameFromEnumValue(ParameterId, Value, DisplayName);
		}
	case FG_BITALIGNMENT:
		switch(Value)
		{
		case FG_LEFT_ALIGNED:
			DisplayName = "Left Aligned";
			return FG_OK;
		case FG_RIGHT_ALIGNED:
			DisplayName = "Right Aligned";
			return FG_OK;
		default:
			return FG_GetEnumNameFromEnumValue(ParameterId, Value, DisplayName);
		}
	default:
		return FG_GetEnumNameFromEnumValue(ParameterId, Value, DisplayName);
	}
}

int CmE5AQ8CXP6D_CameraDevice::FG_GetEnumValueFromDisplayName(int ParameterId, std::string& DisplayName, int *Value)
{
	switch(ParameterId)
	{
	case FG_FORMAT:
		if (strcmp(DisplayName.c_str(), "Gray 8bit") == 0)
		{
			std::string s = "FG_GRAY";
			return FG_GetEnumValueFromEnumName(ParameterId, s, Value);
		}
		return FG_GetEnumValueFromEnumName(ParameterId, DisplayName, Value);
	case FG_BITALIGNMENT:
		if (strcmp(DisplayName.c_str(), "Left Aligned") == 0)
		{
			std::string s = "FG_LEFT_ALIGNED";
			return FG_GetEnumValueFromEnumName(ParameterId, s, Value);
		}
		if (strcmp(DisplayName.c_str(), "Right Aligned") == 0)
		{
			std::string s = "FG_RIGHT_ALIGNED";
			return FG_GetEnumValueFromEnumName(ParameterId, s, Value);
		}
		return FG_GetEnumValueFromEnumName(ParameterId, DisplayName, Value);
	default:
			return FG_GetEnumValueFromEnumName(ParameterId, DisplayName, Value);
	}
}


int CmE5AQ8CXP6D_CameraDevice::Initialize()
{
	int siso_fg_result = FG_OK;		// Errors defined as FG_XXXXX (fgrab_define.h)
	int siso_sgc_result = SGC_OK;	// Errors defined as ERR_SGC_XXXXX (siso_genicam_error.h)
	int mm_device_result = DEVICE_OK;	// Errors defined as DEVICE_XXXXX (MMDeviceConstants.h)
	AddToLog("running CmE5AQ8CXP6D_CameraDevice::Initialize() ");

	// Attention!
	// GetProperty() uses CDeviceUtils::CopyLimitedString(), which copies up to MM::MaxStrLength characters with strncpy().
	// Therefore you HAVE to use MM::MaxStrLength for allocation!
	char selected_camera_string[MM::MaxStrLength] = "";
	// Run-Time Check Failure #2 - Stack around the variable 'selected_camera_string' was corrupted.
	//char selected_camera_string[MM::MaxStrLength-2] = "";

	// SetErrorText()
	char errortext[MM::MaxStrLength];

	char *data_buffer = NULL;
	unsigned int data_buffer_length = 0;

	if (camera_enumeration_error_ != DEVICE_OK)
	{
		return camera_enumeration_error_;
	}
	if (mm_device_initialized_)
	{
		mm_device_result = UpdateStatus(); // ???
		return mm_device_result;
	}
	
	mm_device_result = GetProperty("Camera Selector", selected_camera_string);
	if (mm_device_result != DEVICE_OK)
	{
		return mm_device_result;
	}
	if (strlen(selected_camera_string) == 0)
	{
		sprintf(errortext, "\nCmE5AQ8CXP6D_CameraDevice::Initialize():\n Serial number is required.\n");
		SetErrorText(ERR_SERIAL_NUMBER_REQUIRED, errortext);
		return ERR_SERIAL_NUMBER_REQUIRED;
	}
	if ((strcmp(selected_camera_string, "No Camera found!") == 0) || (camera_device_list_.size() < 1))
	{
		AddToLog("CmE5AQ8CXP6D_CameraDevice::Initialize():\n No Camera found!");
		sprintf(errortext, "\nCmE5AQ8CXP6D_CameraDevice::Initialize():\n No Camera found!\n");
		SetErrorText(ERR_NO_CAMERA_FOUND, errortext);
		return ERR_NO_CAMERA_FOUND;
	}

	siso_fg_result = Fg_InitLibraries(NULL);
	if (siso_fg_result != FG_OK)
	{
		sprintf(errortext, "Initialize():\nFg_InitLibraries() failed: %s", getErrorDescription(siso_fg_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_FG_INITLIBRARIES_FAILED, errortext);
		return ERR_FG_INITLIBRARIES_FAILED;
	}
	siso_libraries_initialized_ = true;

	for (DeviceList_t::iterator it = camera_device_list_.begin(); it != camera_device_list_.end(); it++)
	{
		if (strcmp((*it).CameraSerialNumber.c_str(), selected_camera_string) == 0)
		{
			selected_camera_ = (*it);
			break;
		}
	}
	
	char filename[256];
	sprintf(filename, "%s.mcf", selected_camera_.BoardSerialNumber.c_str());
	
	FILE *fp = fopen(filename, "r");
	if (fp == NULL)
	{
		sprintf(errortext, "Configfile \"%s\" not found.", filename);
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_CONFIGFILE_NOT_FOUND, errortext);
		return ERR_CONFIGFILE_NOT_FOUND;
	}
	fclose(fp);

	framegrabber_ = Fg_InitConfigEx(filename, selected_camera_.BoardIndex, FG_INIT_FLAG_DEFAULT);
	if (framegrabber_ == NULL)
	{
		sprintf(errortext, "Fg_InitConfigEx() failed.\n%s", Fg_getLastErrorDescription(framegrabber_));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}
	framegrabber_initialized_ = true;
	siso_sgc_result = Sgc_initBoard(framegrabber_, 0, &sgc_board_handle_);
	if (siso_sgc_result != SGC_OK)
	{
		sprintf(errortext, "Sgc_initBoard() failed.\n%s", Sgc_getErrorDescription(siso_sgc_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}

	// Auto Link Discovery
	siso_sgc_result = Sgc_scanPorts(sgc_board_handle_, 0xF, 5000, LINK_SPEED_NONE);
	if (siso_sgc_result != SGC_OK)
	{
		sprintf(errortext, "Sgc_scanPorts() failed.\n%s", Sgc_getErrorDescription(siso_sgc_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}

	siso_sgc_result = Sgc_getCameraByIndex(sgc_board_handle_, selected_camera_.PortIndex, &sgc_camera_handle_);
	if (siso_sgc_result != SGC_OK)
	{
		sprintf(errortext, "Sgc_getCameraByIndex() failed.\n%s", Sgc_getErrorDescription(siso_sgc_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}

	siso_sgc_result = Sgc_connectCamera(sgc_camera_handle_);
	if (siso_sgc_result != SGC_OK)
	{
		sprintf(errortext, "Sgc_connectCamera() failed.\n%s", Sgc_getErrorDescription(siso_sgc_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}
	sgc_camera_connected_ = true;
	AddToLog("Camera connected");

	siso_sgc_result = Sgc_LinkConnect(sgc_board_handle_, sgc_camera_handle_);
	if (siso_sgc_result != SGC_OK)
	{
		sprintf(errortext, "Sgc_LinkConnect() failed.\n%s", Sgc_getErrorDescription(siso_sgc_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_INITCONFIG_FAILED, errortext);
		return ERR_INITCONFIG_FAILED;
	}
	sgc_link_connected_ = true;
	AddToLog("link connected");
	try
	{
		genicam_camera_ = new CSiSoGenICamCamera(sgc_camera_handle_, this);
	}
	catch(GenericException &e)
	{
		AddToLog(string(e.GetDescription()));
		SetErrorText(ERR_XML_FROM_CAMERA_FAILED, e.GetDescription());
		return ERR_XML_FROM_CAMERA_FAILED;
	}
	AddToLog("camera is ready");

#if 0
	int fg_num_params = Fg_getNrOfParameter(framegrabber_);
	if (fg_num_params < 0)
	{
		std::string s;
		s.assign(Fg_getLastErrorDescription(framegrabber_));
		assert(0);
	}

	FILE* fp_p = fopen("FgParamNamesIDs.txt", "wt");
	for (int ii = 0; ii < fg_num_params; ii++)
	{
		const char *name = Fg_getParameterName(framegrabber_, ii);
        fprintf(fp_p, " Param %d: %s, 0x%x\n", ii, name, Fg_getParameterId(framegrabber_, ii));
	}
	fclose(fp_p);

	size_t size_needed = 0;
	Fg_getParameterInfoXML(framegrabber_, selected_camera_.PortIndex, NULL, &size_needed);
	char *xml_buffer = (char*)malloc(size_needed);
	Fg_getParameterInfoXML(framegrabber_, selected_camera_.PortIndex, xml_buffer, &size_needed);
	FILE* fp_params = fopen("FgParams.xml", "wt");
	fwrite(xml_buffer, size_needed - 1, 1, fp_params);
	fclose(fp_params);
#endif

	// Initialize MicroManager Device Property Browser

	// MicroManager DeviceName
	mm_device_result = CreateProperty(MM::g_Keyword_Name, g_DeviceName_mE5AQ8CXP6D_CameraDevice, MM::String, true);
	if (DEVICE_OK != mm_device_result)
		return mm_device_result;

#pragma region FrameGrabber Properties

	// FrameGrabber BoardName
	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_BOARDNAME, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if ((siso_fg_result != FG_OK) || (data_buffer_length == 0))
	{
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	data_buffer = (char *)malloc((size_t)data_buffer_length);
	if (data_buffer == NULL)
		return DEVICE_OUT_OF_MEMORY;

	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_BOARDNAME, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if (siso_fg_result != FG_OK)
	{
		free (data_buffer);
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	mm_device_result = CreateProperty("_FG BoardName", data_buffer, MM::String, true);
	if (DEVICE_OK != mm_device_result)
	{
		free(data_buffer);
		return mm_device_result;
	}

	// FrameGrabber SerialNumber
	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_BOARDSERIALNO, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if ((siso_fg_result != FG_OK) || (data_buffer_length == 0))
	{
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	data_buffer = (char *)malloc((size_t)data_buffer_length);
	if (data_buffer == NULL)
		return DEVICE_OUT_OF_MEMORY;
	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_BOARDSERIALNO, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if (siso_fg_result != FG_OK)
	{
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	mm_device_result = CreateProperty("_FG BoardSerialNumber", data_buffer, MM::String, true);
	if (DEVICE_OK != mm_device_result)
		return mm_device_result;

	// FrameGrabber HAP Applet FileName (FG_HapFile)
	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_APPLET_FILE_NAME, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if ((siso_fg_result != FG_OK) || (data_buffer_length == 0))
	{
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	data_buffer = (char *)malloc((size_t)data_buffer_length);
	if (data_buffer == NULL)
		return DEVICE_OUT_OF_MEMORY;
	siso_fg_result = Fg_getSystemInformation(framegrabber_, INFO_APPLET_FILE_NAME, PROP_ID_VALUE, 0, data_buffer, &data_buffer_length);
	if (siso_fg_result != FG_OK)
	{
		free(data_buffer);
		// TODO:
		// provide ErrorDescription
		return ERR_FG_GETSYSTEMINFORMATION_FAILED;
	}
	mm_device_result = CreateProperty("_FG HAP Applet FileName", data_buffer, MM::String, true);
	free(data_buffer);
	data_buffer = NULL;
	data_buffer_length = 0;
	if (DEVICE_OK != mm_device_result)
		return mm_device_result;

	// FG_Width
	int cur_val = 0;
	siso_fg_result = Fg_getParameter(framegrabber_, FG_WIDTH, &cur_val, selected_camera_.PortIndex);
	if (siso_fg_result != FG_OK)
	{
		sprintf(errortext, "Fg_getParameter() failed (FG_WIDTH).\n%s", Fg_getErrorDescription(framegrabber_, siso_fg_result));
		AddToLog(std::string(errortext));
		AddToLog(std::string("\n"));
		SetErrorText(ERR_FG_GET_PARAMETER, errortext);
		return ERR_FG_GET_PARAMETER;
	}
	char strval[128] = "";
	sprintf(strval, "%d", cur_val);
	CPropertyAction* pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_Width);
	mm_device_result = CreateProperty("FG ROI Width", CDeviceUtils::ConvertToString(cur_val), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	// FG_Height
	siso_fg_result = Fg_getParameter(framegrabber_, FG_HEIGHT, &cur_val, selected_camera_.PortIndex);
	if (siso_fg_result != FG_OK)
		assert(0);
	sprintf(strval, "%d", cur_val);
	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_Height);
	mm_device_result = CreateProperty("FG ROI Height", CDeviceUtils::ConvertToString(cur_val), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	// FG_Xoffset
	siso_fg_result = Fg_getParameter(framegrabber_, FG_XOFFSET, &cur_val, selected_camera_.PortIndex);
	if (siso_fg_result != FG_OK)
		assert(0);
	sprintf(strval, "%d", cur_val);
	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_XOffset);
	mm_device_result = CreateProperty("FG ROI XOffset", CDeviceUtils::ConvertToString(cur_val), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	// FG_Yoffset
	siso_fg_result = Fg_getParameter(framegrabber_, FG_YOFFSET, &cur_val, selected_camera_.PortIndex);
	if (siso_fg_result != 0)
		assert(0);
	sprintf(strval, "%d", cur_val);
	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_YOffset);
	mm_device_result = CreateProperty("FG ROI YOffset", CDeviceUtils::ConvertToString(cur_val), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	// FG_Format
	int val;
	std::string str_val;
	siso_fg_result = Fg_getParameter(framegrabber_, FG_FORMAT, &val, selected_camera_.PortIndex);
	if (siso_fg_result != 0)
		assert(0);
	if (FG_GetDisplayNameFromEnumValue(FG_FORMAT, val, str_val) != FG_OK)
		assert(0);

	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_Format);
	mm_device_result = CreateProperty("FG Format", str_val.c_str(), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	Fg_getParameterPropertyEx(framegrabber_, FG_FORMAT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	data_buffer = (char*)malloc(data_buffer_length);
	Fg_getParameterPropertyEx(framegrabber_, FG_FORMAT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	FgPropertyEnumValues* e = (FgPropertyEnumValues*)data_buffer;
	std::vector<std::string> format_enums_grabber;
	while (e->value != -1)
	{
		std::string p;
		if (FG_GetDisplayNameFromEnumValue(FG_FORMAT, e->value, p) != FG_OK)
			assert(0);
		format_enums_grabber.push_back(p);
		e = FG_PROP_GET_NEXT_ENUM_VALUE(e);
	}
	SetAllowedValues("FG Format", format_enums_grabber);

	// FG_Pixeldepth
	siso_fg_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH, &cur_val, selected_camera_.PortIndex);
	if (siso_fg_result != 0)
		assert(0);
	sprintf(strval, "%d", cur_val);
		
	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_PixelDepth);
	mm_device_result = CreateProperty("FG PixelDepth", CDeviceUtils::ConvertToString(cur_val), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	// FG_Bitalignment
	val = 0;
	str_val.clear();
	siso_fg_result = Fg_getParameter(framegrabber_, FG_BITALIGNMENT, &val, selected_camera_.PortIndex);
	if (siso_fg_result != 0)
		assert(0);
	if (FG_GetDisplayNameFromEnumValue(FG_BITALIGNMENT, val, str_val) != FG_OK)
		assert(0);

	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_BitAlignment);
	mm_device_result = CreateProperty("FG BitAlignment", str_val.c_str(), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	Fg_getParameterPropertyEx(framegrabber_, FG_BITALIGNMENT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	data_buffer = (char*)malloc(data_buffer_length);
	Fg_getParameterPropertyEx(framegrabber_, FG_BITALIGNMENT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	e = (FgPropertyEnumValues*)data_buffer;
	std::vector<std::string> bitalignment_enums_grabber;
	while (e->value != -1)
	{
		std::string p;
		if (FG_GetDisplayNameFromEnumValue(FG_BITALIGNMENT, e->value, p) != FG_OK)
			assert(0);
		bitalignment_enums_grabber.push_back(p);
		e = FG_PROP_GET_NEXT_ENUM_VALUE(e);
	}
	SetAllowedValues("FG BitAlignment", bitalignment_enums_grabber);

	// FG_PixelFormat
	val = 0;
	str_val.clear();
	siso_fg_result = Fg_getParameter(framegrabber_, FG_PIXELFORMAT, &val, selected_camera_.PortIndex);
	if (siso_fg_result != 0)
		assert(0);
	if (FG_GetEnumNameFromEnumValue(FG_PIXELFORMAT, val, str_val) != FG_OK)
		assert(0);

	pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_FG_PixelFormat);
	mm_device_result = CreateProperty("FG PixelFormat", str_val.c_str(), MM::String, false, pAct);
	assert(mm_device_result == DEVICE_OK);

	if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}
	Fg_getParameterPropertyEx(framegrabber_, FG_PIXELFORMAT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	data_buffer = (char*)malloc(data_buffer_length);
	Fg_getParameterPropertyEx(framegrabber_, FG_PIXELFORMAT, PROP_ID_ENUM_VALUES, selected_camera_.PortIndex, data_buffer, (int*)(&data_buffer_length));
	e = (FgPropertyEnumValues*)data_buffer;
	std::vector<std::string> pixelformat_enums_grabber;
	while (e->value != -1)
	{
		std::string p = e->name;
		pixelformat_enums_grabber.push_back(p);
		e = FG_PROP_GET_NEXT_ENUM_VALUE(e);
	}
	SetAllowedValues("FG PixelFormat", pixelformat_enums_grabber);

#pragma endregion

#pragma region Camera Properties

	// Properties w/o Property Handler
	AddToLog("reading  Camera Properties");

	// DeviceVendorName
	if (genicam_camera_->IsNodeAvailable("DeviceVendorName"))
	{
		mm_device_result = CreateProperty("CAM DeviceVendorName", genicam_camera_->GetValueAsString("DeviceVendorName").c_str(), MM::String, true);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// DeviceModelName
	if (genicam_camera_->IsNodeAvailable("DeviceModelName"))
	{
		mm_device_result = CreateProperty("CAM DeviceModelName", genicam_camera_->GetValueAsString("DeviceModelName").c_str(), MM::String, true);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// DeviceSerialNumber
	if (genicam_camera_->IsNodeAvailable("DeviceSerialNumber"))
	{
		mm_device_result = CreateProperty("CAM DeviceSerialNumber", genicam_camera_->GetValueAsString("DeviceSerialNumber").c_str(), MM::String, true);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}



	// Properties with Property Handler

	// PixelFormat
	if (genicam_camera_->IsNodeAvailable("PixelFormat"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_PixelFormat);
		mm_device_result = CreateProperty("CAM PixelFormat", genicam_camera_->GetValueAsString("PixelFormat").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// Width
	if (genicam_camera_->IsNodeAvailable("Width"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_Width);
		mm_device_result = CreateProperty("CAM ROI Width", genicam_camera_->GetValueAsString("Width").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// Height
	if (genicam_camera_->IsNodeAvailable("Height"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_Height);
		mm_device_result = CreateProperty("CAM ROI Height", genicam_camera_->GetValueAsString("Height").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}


	// OffsetX
	if (genicam_camera_->IsNodeAvailable("OffsetX"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_OffsetX);
		mm_device_result = CreateProperty("CAM ROI XOffset", genicam_camera_->GetValueAsString("OffsetX").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// OffsetY
	if (genicam_camera_->IsNodeAvailable("OffsetY"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_OffsetY);
		mm_device_result = CreateProperty("CAM ROI YOffset", genicam_camera_->GetValueAsString("OffsetY").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// ExposureMode
	if (genicam_camera_->IsNodeAvailable("ExposureMode"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureMode);
		mm_device_result = CreateProperty("CAM ExposureMode", genicam_camera_->GetValueAsString("ExposureMode").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// ExposureTime (IFloat)
	// or
	// ExposureTimeRaw (IInteger)
	if (genicam_camera_->IsNodeAvailable("ExposureTime"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureTime);
		mm_device_result = CreateProperty("CAM ExposureTime", genicam_camera_->GetValueAsString("ExposureTime").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}
	else if (genicam_camera_->IsNodeAvailable("ExposureTimeRaw"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureTimeRaw);
		mm_device_result = CreateProperty("CAM ExposureTimeRaw", genicam_camera_->GetValueAsString("ExposureTimeRaw").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}
	else
		assert(0);


	// BinningHorizontal
	// BinningVertical

	// BinningMode (IEnumeration)
	if (genicam_camera_->IsNodeAvailable("BinningMode"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_BinningMode);
		mm_device_result = CreateProperty("CAM BinningMode", genicam_camera_->GetValueAsString("BinningMode").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// ReverseX (IBoolean)
	if (genicam_camera_->IsNodeAvailable("ReverseX"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_ReverseX);
		mm_device_result = CreateProperty("CAM ReverseX", genicam_camera_->GetValueAsString("ReverseX").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
		ClearAllowedValues("CAM ReverseX");
		AddAllowedValue("CAM ReverseX", "0");
		AddAllowedValue("CAM ReverseX", "1");
	}

	// (ReverseY) (IBoolean)
	if (genicam_camera_->IsNodeAvailable("ReverseY"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_ReverseY);
		mm_device_result = CreateProperty("CAM ReverseY", genicam_camera_->GetValueAsString("ReverseY").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
		ClearAllowedValues("CAM ReverseY");
		AddAllowedValue("CAM ReverseY", "0");
		AddAllowedValue("CAM ReverseY", "1");
	}

	// TestImageSelector (IEnumeration)
	if (genicam_camera_->IsNodeAvailable("TestImageSelector"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_TestImageSelector);
		mm_device_result = CreateProperty("CAM TestImageSelector", genicam_camera_->GetValueAsString("TestImageSelector").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// TestImageVideoLevel (IInteger)
	if (genicam_camera_->IsNodeAvailable("TestImageVideoLevel"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_TestImageVideoLevel);
		mm_device_result = CreateIntegerProperty("CAM TestImageVideoLevel", (long)genicam_camera_->GetIntegerValue("TestImageVideoLevel"), false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// CrosshairOverlay (IBoolean)
	if (genicam_camera_->IsNodeAvailable("CrosshairOverlay"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_CrosshairOverlay);
		mm_device_result = CreateStringProperty("CAM CrosshairOverlay", genicam_camera_->GetValueAsString("CrosshairOverlay").c_str(), false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
		ClearAllowedValues("CAM CrosshairOverlay");
		AddAllowedValue("CAM CrosshairOverlay", "0");
		AddAllowedValue("CAM CrosshairOverlay", "1");
	}

	// AcquisitionFrameRate (IFloat)
	if (genicam_camera_->IsNodeAvailable("AcquisitionFrameRate"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionFrameRate);
		mm_device_result = CreateProperty("CAM AcquisitionFrameRate", genicam_camera_->GetValueAsString("AcquisitionFrameRate").c_str(), MM::String, false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}

	// AcquisitionFramePeriod
	// AcquisitionFramePeriodRaw

	// AcquisitionMaxFrameRate (ICommand)
	if (genicam_camera_->IsNodeAvailable("AcquisitionMaxFrameRate"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionMaxFrameRate);
		mm_device_result = CreateStringProperty("CAM AcquisitionMaxFrameRate", "0", false, pAct);
		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
		ClearAllowedValues("CAM AcquisitionMaxFrameRate");
		AddAllowedValue("CAM AcquisitionMaxFrameRate", "0");
		AddAllowedValue("CAM AcquisitionMaxFrameRate", "1");
	}

	// TriggerSource (IEnumeration)
	if (genicam_camera_->IsNodeAvailable("TriggerSource"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_TriggerSource);
		mm_device_result = CreateProperty("CAM TriggerSource", genicam_camera_->GetValueAsString("TriggerSource").c_str(), MM::String, false, pAct);

		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}
		// TriggerActivation (IEnumeration)
	if (genicam_camera_->IsNodeAvailable("TriggerActivation"))
	{
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_TriggerActivation);
		mm_device_result = CreateProperty("CAM TriggerActivation", genicam_camera_->GetValueAsString("TriggerActivation").c_str(), MM::String, false, pAct);

		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}
		// AcquisitionMode (IEnumeration)
	if (genicam_camera_->IsNodeAvailable("AcquisitionMode"))
	{
		// some camera manufatures are using AcquisitionMode "single " to trigger the camera externaly(like trigger source).
		pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionMode);
		mm_device_result = CreateProperty("CAM AcquisitionMode", genicam_camera_->GetValueAsString("AcquisitionMode").c_str(), MM::String, false, pAct);

		if (mm_device_result != DEVICE_OK)
			return mm_device_result;
	}


	// GainSelector
	// Gain
	// GainRaw
	// BlackLevel
	// BlackLevelRaw
	// UserSetSelector
	// UserSetLoad
	// UserSetSave
	// SensorTemperature


#pragma endregion


   // Keyword_Binning
   pAct = new CPropertyAction(this, &CmE5AQ8CXP6D_CameraDevice::OnBinning);
   mm_device_result = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
   if (DEVICE_OK != mm_device_result)
	   return mm_device_result;
   SetPropertyLimits(MM::g_Keyword_Binning, 1, 1);

    if (data_buffer != NULL)
	{
		free(data_buffer);
		data_buffer = NULL;
		data_buffer_length = 0;
	}

   mm_device_initialized_ = true;
   AddToLog("leaving CmE5AQ8CXP6D_CameraDevice::Initialize() ");
   return DEVICE_OK;
}

void CmE5AQ8CXP6D_CameraDevice::AddToLog(std::string msg)
{
	LogMessage(msg, false);
}

int CmE5AQ8CXP6D_CameraDevice::Shutdown()
{
	int siso_result = DEVICE_OK;

	mm_device_initialized_ = false;

	if (genicam_camera_ != NULL)
	{
		delete genicam_camera_;
		genicam_camera_ = NULL;
	}

	if (sgc_link_connected_)
	{
		siso_result = Sgc_LinkDisconnect(sgc_camera_handle_);
		if (siso_result != DEVICE_OK)
			return siso_result;
		sgc_link_connected_ = false;
	}

	if (sgc_camera_connected_)
	{
		siso_result = Sgc_disconnectCamera(sgc_camera_handle_);
		if (siso_result != DEVICE_OK)
			return siso_result;
		sgc_camera_connected_ = false;
	}

	if (sgc_board_initialized_)
	{
		Sgc_freeBoard(sgc_board_handle_);
		sgc_board_initialized_ = false;
	}

	if (framegrabber_initialized_)
	{
		Fg_FreeGrabber(framegrabber_);
		framegrabber_initialized_ = false;
	}

	if (siso_libraries_initialized_)
	{
		Fg_FreeLibraries();
		siso_libraries_initialized_ = false;
	}

	if (snap_buffer_ != NULL)
	{
		free(snap_buffer_);
		snap_buffer_ = NULL;
		snap_buffer_size_ = 0;
	}

   return DEVICE_OK;
}

DeviceList_t CmE5AQ8CXP6D_CameraDevice::EnumerateCameras()
{
	int siso_result = DEVICE_OK;

	Shutdown();

	siso_result = Fg_InitLibraries(NULL);

	char buffer[256];
	unsigned int buflen = sizeof(buffer);
	buffer[0] = 0;

	// availability : starting with RT 5.2
	siso_result = Fg_getSystemInformation(NULL, INFO_NR_OF_BOARDS, PROP_ID_VALUE, 0, buffer, &buflen);
	int nrOfBoards = atoi(buffer);
	DeviceList_t devices;
	for (int i = 0; i < nrOfBoards; i++)
	{
		buffer[0]= 0;
		buflen = sizeof(buffer);
		siso_result = Fg_getSystemInformation(NULL, INFO_BOARDTYPE, PROP_ID_VALUE, i, buffer, &buflen);
		int board_type = atoi(buffer);
		if (board_type == PN_MICROENABLE5AQ8CXP6D)
		{
			DeviceListEntry_t entry;
			entry.BoardIndex = i;
			entry.BoardType = board_type;
			buffer[0] = 0;
			buflen = sizeof(buffer);
			siso_result = Fg_getSystemInformation(NULL, INFO_BOARDSERIALNO, PROP_ID_VALUE, i, buffer, &buflen);
			entry.BoardSerialNumber.assign(buffer);

			char filename[256];
			sprintf(filename, "%s.mcf", entry.BoardSerialNumber.c_str());
			Fg_Struct* fg = Fg_InitConfigEx(filename, i, FG_INIT_FLAG_DEFAULT);

			if (fg == NULL)
			{
				// ToDo: Error handling - FrameGrabber might be in use by another application!!!
				assert(0);
			}
			else
			{
				SgcBoardHandle *sgc_board_handle = NULL;
				siso_result = Sgc_initBoard(fg, 0, &sgc_board_handle);
				siso_result = Sgc_scanPorts(sgc_board_handle, 0xF, 5000, LINK_SPEED_NONE);
				int num_cam_ports = Sgc_getCameraCount(sgc_board_handle);
					
				for (unsigned int cameraIndex = 0; cameraIndex < (unsigned int)num_cam_ports; cameraIndex++)
				{
					SgcCameraHandle* ch = NULL;
					siso_result = Sgc_getCameraByIndex(sgc_board_handle,cameraIndex, &ch);
					if (siso_result == DEVICE_OK)
					{
						entry.PortIndex = cameraIndex;
						SgcCameraInfo* cam_info = Sgc_getCameraInfo(ch);
						entry.CameraSerialNumber.assign(cam_info->deviceSerialNumber);
						entry.CameraModelName.assign(cam_info->deviceModelName);
						devices.push_back(entry);
					}
				}
				Sgc_freeBoard(sgc_board_handle);
				Fg_FreeGrabber(fg);
			}
		}
	}
	Fg_FreeLibraries();
	return devices;
}


void CmE5AQ8CXP6D_CameraDevice::ResizeSnapBuffer()
{
	if (snap_buffer_ != NULL)
		free(snap_buffer_);
	long bytes = GetImageBufferSize();
	snap_buffer_ = malloc(bytes);
	snap_buffer_size_ = bytes;
}


// CCameraBase<U>
// --------------

int CmE5AQ8CXP6D_CameraDevice::SnapImage()
{
	const unsigned int c_MemoryBuffers(1);
	dma_mem *memhdr = NULL;
	memhdr = Fg_AllocMemEx(framegrabber_, GetImageBufferSize() * c_MemoryBuffers, c_MemoryBuffers);
	if (memhdr == NULL)
		return DEVICE_SNAP_IMAGE_FAILED;

	ResizeSnapBuffer();

	int siso_result = Sgc_executeCommand(sgc_camera_handle_, "AcquisitionStart");
	siso_result = Fg_AcquireEx(framegrabber_, selected_camera_.PortIndex, 1, ACQ_STANDARD, memhdr);
	frameindex_t pic_no = Fg_getLastPicNumberBlockingEx(framegrabber_, 1, selected_camera_.PortIndex, 5000, memhdr);
	siso_result = Sgc_executeCommand(sgc_camera_handle_, "AcquisitionStop");
	void * image_ptr = Fg_getImagePtrEx(framegrabber_, pic_no, selected_camera_.PortIndex, memhdr);
	memcpy(snap_buffer_, image_ptr, snap_buffer_size_);
	Fg_FreeMemEx(framegrabber_, memhdr);


#if 0
		camera_->StartGrabbing(1, GrabStrategy_OneByOne, GrabLoop_ProvidedByUser);
		// This smart pointer will receive the grab result data.
		//When all smart pointers referencing a Grab Result Data object go out of scope, the grab result's image buffer is reused for grabbing
		CGrabResultPtr ptrGrabResult;
		int timeout_ms = 5000;
		if (!camera_->RetrieveResult(timeout_ms, ptrGrabResult, TimeoutHandling_ThrowException)) {
			return DEVICE_ERR;
		}
		if (!ptrGrabResult->GrabSucceeded()) {
			return DEVICE_ERR;
		}
		if (ptrGrabResult->GetPayloadSize() != imgBufferSize_)
		{// due to parameter change on  binning
			ResizeSnapBuffer();
		}
		CopyToImageBuffer(ptrGrabResult);

	}
	catch (const GenericException & e)
	{
		// Error handling.
		AddToLog(e.GetDescription());
		cerr << "An exception occurred." << endl
			<< e.GetDescription() << endl;
	}
#endif
	return DEVICE_OK;
}

unsigned char const* CmE5AQ8CXP6D_CameraDevice::GetImageBuffer()
{
	return (unsigned char*)snap_buffer_;
}

unsigned int CmE5AQ8CXP6D_CameraDevice::GetImageWidth() const
{
	int value = 0;
	int siso_fg_result = Fg_getParameter(framegrabber_, FG_WIDTH, &value, selected_camera_.PortIndex);
	siso_fg_result; // eleminate warning
	return (unsigned int)value;
}

unsigned int CmE5AQ8CXP6D_CameraDevice::GetImageHeight() const
{
	int value;
	int siso_fg_result = Fg_getParameter(framegrabber_, FG_HEIGHT, &value, selected_camera_.PortIndex);
	siso_fg_result; // eleminate warning
	return (unsigned int)value;
}

unsigned int CmE5AQ8CXP6D_CameraDevice::GetImageBytesPerPixel() const
{
	int value = 0;
	int siso_fg_result = Fg_getParameter(framegrabber_, FG_FORMAT, &value, selected_camera_.PortIndex);
	siso_fg_result; // eleminate warning
	switch (value)
	{
	case FG_GRAY:
		value = 1;
		break;
	case FG_GRAY16:
		value = 2;
		break;
	case FG_COL24:
		value = 3;
		break;
	case FG_COL32:
	case FG_RGBX32:
		value = 4;
		break;
	default:
		assert(0);
	}
	return value;
}


// MM::Camera
// ----------

long CmE5AQ8CXP6D_CameraDevice::GetImageBufferSize() const
{
	long size = GetImageWidth() * GetImageHeight() * GetImageBytesPerPixel();
	return size;
}

unsigned int CmE5AQ8CXP6D_CameraDevice::GetBitDepth() const
{
#ifdef BITDEPTH_FROM_CAMERA
	assert(0);
	//
#else
	int value = 0;
	Fg_getParameter(framegrabber_, FG_PIXELDEPTH, &value, selected_camera_.PortIndex);
	return value;
#endif
}

int CmE5AQ8CXP6D_CameraDevice::GetBinning() const
{
	//
	assert(0 && "GetBinning!\n");
	return 0;
}

int CmE5AQ8CXP6D_CameraDevice::SetBinning(int binSize)
{
	binSize;
	return DEVICE_NOT_YET_IMPLEMENTED;
}

void CmE5AQ8CXP6D_CameraDevice::SetExposure(double exp)
{
	if (genicam_camera_->IsNodeAvailable("ExposureTime"))
	{
		// from ms to us
		double exp_us = exp * 1000.0;
		genicam_camera_->SetFloatValue("ExposureTime", exp_us);
		return;
	}
	else if (genicam_camera_->IsNodeAvailable("ExposureTimeRaw"))
	{
		// from ms to us
		int exp_us = (int)(exp * 1000.0);
		genicam_camera_->SetIntegerValue("ExposureTimeRaw", exp_us);
	}
	assert(0);
}

double CmE5AQ8CXP6D_CameraDevice::GetExposure() const
{
	if (genicam_camera_->IsNodeAvailable("ExposureTime"))
	{
		// from us to ms
		double exp_ms = genicam_camera_->GetFloatValue("ExposureTime") / 1000.0;
		return exp_ms;
	}
	else if (genicam_camera_->IsNodeAvailable("ExposureTimeRaw"))
	{
		// from us to ms
		double exp_ms = (double)genicam_camera_->GetIntegerValue("ExposureTimeRaw") / 1000.0;
		return exp_ms;
	}
	assert(0);
	return 0.0;
}

int CmE5AQ8CXP6D_CameraDevice::SetROI(unsigned int x, unsigned int y, unsigned int xSize, unsigned int ySize)
{
	x;
	y;
	xSize;
	ySize;
	return DEVICE_NOT_YET_IMPLEMENTED;
}

int CmE5AQ8CXP6D_CameraDevice::GetROI(unsigned int& x, unsigned int& y, unsigned int& xSize, unsigned int& ySize)
{
	int siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET, &x, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET, &y, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_WIDTH, &xSize, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT, &ySize, selected_camera_.PortIndex);
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::ClearROI()
{
	//
	return DEVICE_NOT_YET_IMPLEMENTED;
}


// Property Handlers
//

#if 0
   enum ActionType {
      NoAction,
      BeforeGet,
      AfterSet,
      IsSequenceable,
      AfterLoadSequence,
      StartSequence,
      StopSequence
   };

/**
 * \brief Definitions of virtual parameter IDs
 * These values can be used to get additional information about a parameter by adding them to the parameter ID and call one of the
 * Fg_getParameter... functions.
 */
#define FG_PARAMETER_PROPERTY_ACCESS 0x80000000 /**< Get the access mode of a parameter by adding this value to the parameter ID; this is always an int parameter! */
#define FG_PARAMETER_PROPERTY_MIN    0xC0000000 /**< Get the minimum value of a parameter by adding this value to the parameter ID; same type as parameter */
#define FG_PARAMETER_PROPERTY_MAX    0x40000000 /**< Get the maximum value of a parameter by adding this value to the parameter ID; same type as parameter */
#define FG_PARAMETER_PROPERTY_STEP   0xE0000000 /**< Get the step value of a parameter by adding this value to the parameter ID; same type as parameter */

/**
 * \brief Possible flags for parameter access property
 * The value of the access property for a parameter can be any combination of the values below ORed
 */
#define FP_PARAMETER_PROPERTY_ACCESS_READ   0x1 /**< The register can be accessed for reading */
#define FP_PARAMETER_PROPERTY_ACCESS_WRITE  0x2 /**< The register can be accessed for writing */
#define FP_PARAMETER_PROPERTY_ACCESS_MODIFY 0x4 /**< The register value can be modified during acquisition  */
#define FP_PARAMETER_PROPERTY_ACCESS_LOCKED 0x8 /**< The register value is locked */

#endif

int CmE5AQ8CXP6D_CameraDevice::On_FG_Width(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	int value = 0;
	int value_min = 0;
	int value_max = 0;
	int value_step = 0;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_WIDTH + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_WIDTH + FG_PARAMETER_PROPERTY_MIN, &value_min, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_WIDTH + FG_PARAMETER_PROPERTY_MAX, &value_max, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_WIDTH + FG_PARAMETER_PROPERTY_STEP, &value_step, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_WIDTH, &value, selected_camera_.PortIndex);
		pProp->Set((long)value);
	}
	else if (pAct == MM::AfterSet)
	{
		std::string strval;
		pProp->Get(strval);
		value = atoi(strval.c_str());
		value = value - (value % value_step);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;
		siso_result = Fg_setParameter(framegrabber_, FG_WIDTH, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value != atoi(strval.c_str()))
			pProp->Set((long)value);
	}
	pProp->SetLimits((double)value_min, (double)value_max);
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_Height(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	int value = 0;
	int value_min = 0;
	int value_max = 0;
	int value_step = 0;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT + FG_PARAMETER_PROPERTY_MIN, &value_min, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT + FG_PARAMETER_PROPERTY_MAX, &value_max, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT + FG_PARAMETER_PROPERTY_STEP, &value_step, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_HEIGHT, &value, selected_camera_.PortIndex);
		pProp->Set((long)value);
	}
	else if (pAct == MM::AfterSet)
	{
		std::string strval;
		pProp->Get(strval);
		value = atoi(strval.c_str());
		value = value - (value % value_step);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;
		siso_result = Fg_setParameter(framegrabber_, FG_HEIGHT, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value != atoi(strval.c_str()))
			pProp->Set((long)value);
	}
	pProp->SetLimits((double)value_min, (double)value_max);
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_XOffset(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	int value = 0;
	int value_min = 0;
	int value_max = 0;
	int value_step = 0;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET + FG_PARAMETER_PROPERTY_MIN, &value_min, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET + FG_PARAMETER_PROPERTY_MAX, &value_max, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET + FG_PARAMETER_PROPERTY_STEP, &value_step, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_XOFFSET, &value, selected_camera_.PortIndex);
		pProp->Set((long)value);

	}
	else if (pAct == MM::AfterSet)
	{
		std::string strval;
		pProp->Get(strval);
		value = atoi(strval.c_str());
		value = value - (value % value_step);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;
		siso_result = Fg_setParameter(framegrabber_, FG_XOFFSET, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value != atoi(strval.c_str()))
			pProp->Set((long)value);
	}
	pProp->SetLimits((double)value_min, (double)value_max);
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_YOffset(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	int value = 0;
	int value_min = 0;
	int value_max = 0;
	int value_step = 0;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET + FG_PARAMETER_PROPERTY_MIN, &value_min, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET + FG_PARAMETER_PROPERTY_MAX, &value_max, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET + FG_PARAMETER_PROPERTY_STEP, &value_step, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_YOFFSET, &value, selected_camera_.PortIndex);
		pProp->Set((long)value);

	}
	else if (pAct == MM::AfterSet)
	{
		std::string strval;
		pProp->Get(strval);
		value = atoi(strval.c_str());
		value = value - (value % value_step);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;
		siso_result = Fg_setParameter(framegrabber_, FG_YOFFSET, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value != atoi(strval.c_str()))
			pProp->Set((long)value);
	}
	pProp->SetLimits((double)value_min, (double)value_max);
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_Format(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	int value;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_FORMAT + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_FORMAT, &value, selected_camera_.PortIndex);
		siso_result = FG_GetDisplayNameFromEnumValue(FG_FORMAT, value, str_value);
		assert(siso_result == FG_OK);
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		siso_result = FG_GetEnumValueFromDisplayName(FG_FORMAT, str_value, &value);
		siso_result = Fg_setParameter(framegrabber_, FG_FORMAT, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_PixelDepth(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	int value = 0;
	int value_min = 0;
	int value_max = 0;
	int value_step = 0;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH + FG_PARAMETER_PROPERTY_MIN, &value_min, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH + FG_PARAMETER_PROPERTY_MAX, &value_max, selected_camera_.PortIndex);
	siso_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH + FG_PARAMETER_PROPERTY_STEP, &value_step, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_PIXELDEPTH, &value, selected_camera_.PortIndex);
		pProp->Set((long)value);

	}
	else if (pAct == MM::AfterSet)
	{
		std::string strval;
		pProp->Get(strval);
		value = atoi(strval.c_str());
		value = value - (value % value_step);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;
		siso_result = Fg_setParameter(framegrabber_, FG_PIXELDEPTH, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		if (value != atoi(strval.c_str()))
			pProp->Set((long)value);
	}
	pProp->SetLimits((double)value_min, (double)value_max);
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_BitAlignment(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	int value;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_BITALIGNMENT + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_BITALIGNMENT, &value, selected_camera_.PortIndex);
		siso_result = FG_GetDisplayNameFromEnumValue(FG_BITALIGNMENT, value, str_value);
		assert(siso_result == FG_OK);
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		siso_result = FG_GetEnumValueFromDisplayName(FG_BITALIGNMENT, str_value, &value);
		siso_result = Fg_setParameter(framegrabber_, FG_BITALIGNMENT, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_FG_PixelFormat(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	int value;
	int value_access = 0;

	int siso_result = Fg_getParameter(framegrabber_, FG_PIXELFORMAT + FG_PARAMETER_PROPERTY_ACCESS, &value_access, selected_camera_.PortIndex);
	
	if (pAct == MM::BeforeGet)
	{
		if((value_access & FP_PARAMETER_PROPERTY_ACCESS_READ) == 0)
			return DEVICE_INVALID_PROPERTY;

		siso_result = Fg_getParameter(framegrabber_, FG_PIXELFORMAT, &value, selected_camera_.PortIndex);
		siso_result = FG_GetEnumNameFromEnumValue(FG_PIXELFORMAT, value, str_value);
		assert(siso_result == FG_OK);
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if ((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0)
			return DEVICE_CAN_NOT_SET_PROPERTY;
		siso_result = FG_GetEnumValueFromEnumName(FG_PIXELFORMAT, str_value, &value);
		siso_result = Fg_setParameter(framegrabber_, FG_PIXELFORMAT, &value, selected_camera_.PortIndex);
		if (siso_result != DEVICE_OK)
			return DEVICE_CAN_NOT_SET_PROPERTY;
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly((value_access & FP_PARAMETER_PROPERTY_ACCESS_WRITE) == 0);

	return DEVICE_OK;
}






int CmE5AQ8CXP6D_CameraDevice::On_CAM_PixelFormat(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	using namespace GenApi;

	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("PixelFormat");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		str_value = genicam_camera_->GetValueAsString("PixelFormat");
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("PixelFormat", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("PixelFormat");
		SetAllowedValues("CAM PixelFormat", list);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_Width(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("Width"));
	CIntegerPtr ptr_width(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;
		string ss = string(ptr_width->ToString().c_str());
		pProp->Set(ss.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		std::string str_value;
		pProp->Get(str_value);
		int width = atoi(str_value.c_str());
		int width_min = (int)ptr_width->GetMin();
		int width_max = (int)ptr_width->GetMax();
		int width_inc = (int)ptr_width->GetInc();
		width = width - (width % width_inc);
		if (width < width_min)
			width = width_min;
		if (width > width_max)
			width = width_max;

		ptr_width->SetValue(width);
		int width_org = atoi(str_value.c_str());
		if (width != width_org)
			pProp->Set((long)width);
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_width->GetMin(), (double)ptr_width->GetMax());
	}
	return DEVICE_OK;
}


int CmE5AQ8CXP6D_CameraDevice::On_CAM_Height(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("Height"));
	CIntegerPtr ptr_height(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;

		pProp->Set(ptr_height->ToString().c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		std::string str_value;
		pProp->Get(str_value);
		int height = atoi(str_value.c_str());
		int height_min = (int)ptr_height->GetMin();
		int height_max = (int)ptr_height->GetMax();
		int height_inc = (int)ptr_height->GetInc();
		height = height - (height % height_inc);
		if (height < height_min)
			height = height_min;
		if (height > height_max)
			height = height_max;

		ptr_height->SetValue(height);
		if (height != atoi(str_value.c_str()))
			pProp->Set((long)height);
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_height->GetMin(), (double)ptr_height->GetMax());
	}
	return DEVICE_OK;
}


int CmE5AQ8CXP6D_CameraDevice::On_CAM_OffsetX(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("OffsetX"));
	CIntegerPtr ptr_offsetx(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;

		pProp->Set(ptr_offsetx->ToString().c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		std::string str_value;
		pProp->Get(str_value);
		int offsetx = atoi(str_value.c_str());
		int offsetx_min = (int)ptr_offsetx->GetMin();
		int offsetx_max = (int)ptr_offsetx->GetMax();
		int offsetx_inc = (int)ptr_offsetx->GetInc();
		offsetx = offsetx - (offsetx % offsetx_inc);
		if (offsetx < offsetx_min)
			offsetx = offsetx_min;
		if (offsetx > offsetx_max)
			offsetx = offsetx_max;

		ptr_offsetx->SetValue(offsetx);
		if (offsetx != atoi(str_value.c_str()))
			pProp->Set((long)offsetx);
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_offsetx->GetMin(), (double)ptr_offsetx->GetMax());
	}
	return DEVICE_OK;
}


int CmE5AQ8CXP6D_CameraDevice::On_CAM_OffsetY(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("OffsetY"));
	CIntegerPtr ptr_offsety(ptr_node);

	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;

		pProp->Set(genicam_camera_->GetValueAsString("OffsetY").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		std::string str_value;
		pProp->Get(str_value);
		int offsety = atoi(str_value.c_str());
		int offsety_min = (int)ptr_offsety->GetMin();
		int offsety_max = (int)ptr_offsety->GetMax();
		int offsety_inc = (int)ptr_offsety->GetInc();
		offsety = offsety - (offsety % offsety_inc);
		if (offsety < offsety_min)
			offsety = offsety_min;
		if (offsety > offsety_max)
			offsety = offsety_max;

		ptr_offsety->SetValue(offsety);
		if (offsety != atoi(str_value.c_str()))
			pProp->Set((long)offsety);
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_offsety->GetMin(), (double)ptr_offsety->GetMax());
	}
	return DEVICE_OK;
}








int CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureTime(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("ExposureTime"));
	CFloatPtr ptr_float(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;
		double value = ptr_float->GetValue();
		pProp->Set(CDeviceUtils::ConvertToString(value));
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		double value, value_org;
		pProp->Get(value_org);
		value = value_org;
		double value_min = ptr_float->GetMin();
		double value_max = ptr_float->GetMax();
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;

		ptr_float->SetValue(value);
		if (value != value_org)
			pProp->Set(CDeviceUtils::ConvertToString(value));
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_float->GetMin(), (double)ptr_float->GetMax());
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureTimeRaw(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("ExposureTimeRaw"));
	CIntegerPtr ptr_value(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;

		pProp->Set((long)ptr_value->GetValue());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		int value, value_org;
		pProp->Get((long&)value_org);
		value = value_org;
		int value_min = (int)ptr_value->GetMin();
		int value_max = (int)ptr_value->GetMax();
		int value_inc = (int)ptr_value->GetInc();
		value = value - (value % value_inc);
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;

		ptr_value->SetValue(value);
		if (value != value_org)
			pProp->Set((long)value);
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_value->GetMin(), (double)ptr_value->GetMax());
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_ExposureMode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("ExposureMode");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		str_value = genicam_camera_->GetValueAsString("ExposureMode");
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("ExposureMode", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("ExposureMode");
		SetAllowedValues("CAM ExposureMode", list);
	}
	return DEVICE_OK;
}






int CmE5AQ8CXP6D_CameraDevice::On_CAM_BinningMode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	using namespace GenApi;

	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("BinningMode");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		pProp->Set(genicam_camera_->GetValueAsString("BinningMode").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("BinningMode", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("BinningMode");
		SetAllowedValues("CAM BinningMode", list);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_ReverseX(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("ReverseX"));
	
	if (pAct == MM::BeforeGet)
	{
		if(!genicam_camera_->IsNodeReadable("ReverseX"))
		{
			return ERR_NODE_NOT_READABLE;
		}
		pProp->Set(genicam_camera_->GetValueAsString("ReverseX").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("ReverseX"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		pProp->Get(str_value);
		genicam_camera_->SetValueFromString("ReverseX", str_value);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_ReverseY(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("ReverseY"));
	
	if (pAct == MM::BeforeGet)
	{
		if(!genicam_camera_->IsNodeReadable("ReverseY"))
		{
			return ERR_NODE_NOT_READABLE;
		}
		pProp->Set(genicam_camera_->GetValueAsString("ReverseY").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("ReverseY"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		pProp->Get(str_value);
		genicam_camera_->SetValueFromString("ReverseY", str_value);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_TestImageSelector(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		if(!genicam_camera_->IsNodeReadable("TestImageSelector"))
		{
			return ERR_NODE_NOT_READABLE;
		}
		pProp->Set(genicam_camera_->GetValueAsString("TestImageSelector").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("TestImageSelector"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		std::string str_value;
		pProp->Get(str_value);
		genicam_camera_->SetValueFromString("TestImageSelector", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("TestImageSelector"));
	if (genicam_camera_->IsNodeReadable("TestImageSelector"))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("TestImageSelector");
		SetAllowedValues("CAM TestImageSelector", list);
	}
	return DEVICE_OK;
}


int CmE5AQ8CXP6D_CameraDevice::On_CAM_TestImageVideoLevel(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		if(!genicam_camera_->IsNodeReadable("TestImageVideoLevel"))
		{
			return ERR_NODE_NOT_READABLE;
		}
		pProp->Set(genicam_camera_->GetValueAsString("TestImageVideoLevel").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("TestImageVideoLevel"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		std::string str_value;
		pProp->Get(str_value);
		try
		{
			genicam_camera_->SetValueFromString("TestImageVideoLevel", str_value);
		}
		catch (GenericException e)
		{
			stringstream ss;
			ss << "\nNode is not writable.\n" << e.what();
			SetErrorText(ERR_NODE_NOT_WRITABLE, ss.str().c_str());
			return ERR_NODE_NOT_WRITABLE;
		}
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("TestImageVideoLevel"));
	if (genicam_camera_->IsNodeReadable("TestImageVideoLevel"))
	{
		CIntegerPtr ptrInteger(genicam_camera_->GetNodeMap()->GetNode("TestImageVideoLevel"));
		SetPropertyLimits(pProp->GetName().c_str(), (double)ptrInteger->GetMin(), (double)ptrInteger->GetMax());
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_CrosshairOverlay(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("CrosshairOverlay"));
	
	if (pAct == MM::BeforeGet)
	{
		if(!genicam_camera_->IsNodeReadable("CrosshairOverlay"))
		{
			return ERR_NODE_NOT_READABLE;
		}
		pProp->Set(genicam_camera_->GetValueAsString("CrosshairOverlay").c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("CrosshairOverlay"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		pProp->Get(str_value);
		genicam_camera_->SetValueFromString("CrosshairOverlay", str_value);
	}
	return DEVICE_OK;
}


int CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionFrameRate(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	INode *ptr_node(genicam_camera_->GetNodeMap()->GetNode("AcquisitionFrameRate"));
	CFloatPtr ptr_float(ptr_node);
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(ptr_node))
			return DEVICE_INVALID_PROPERTY;
		double value = ptr_float->GetValue();
		pProp->Set(CDeviceUtils::ConvertToString(value));
	}
	else if (pAct == MM::AfterSet)
	{
		if (!IsWritable(ptr_node))
			return DEVICE_CAN_NOT_SET_PROPERTY;

		double value, value_org;
		pProp->Get(value_org);
		value = value_org;
		double value_min = ptr_float->GetMin();
		double value_max = ptr_float->GetMax();
		if (value < value_min)
			value = value_min;
		if (value > value_max)
			value = value_max;

		ptr_float->SetValue(value);
		if (value != value_org)
			pProp->Set(CDeviceUtils::ConvertToString(value));
	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(ptr_node));
	if (IsReadable(ptr_node))
	{
		pProp->SetLimits((double)ptr_float->GetMin(), (double)ptr_float->GetMax());
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_TriggerSource(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("TriggerSource");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		str_value = genicam_camera_->GetValueAsString("TriggerSource");
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("TriggerSource", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("TriggerSource");
		SetAllowedValues("CAM TriggerSource", list);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_TriggerActivation(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("TriggerActivation");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		str_value = genicam_camera_->GetValueAsString("TriggerActivation");
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("TriggerActivation", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("TriggerActivation");
		SetAllowedValues("CAM TriggerActivation", list);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionMode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;

	INodeMap *node_map = genicam_camera_->GetNodeMap();
	INode *node = node_map->GetNode("AcquisitionMode");
	
	if (pAct == MM::BeforeGet)
	{
		if(!IsReadable(node))
			return DEVICE_INVALID_PROPERTY;

		str_value = genicam_camera_->GetValueAsString("AcquisitionMode");
		pProp->Set(str_value.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(str_value);
		if (!IsWritable(node))
			return DEVICE_CAN_NOT_SET_PROPERTY;
		genicam_camera_->SetValueFromString("AcquisitionMode", str_value);

	}
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!IsWritable(node));
	if (IsReadable(node))
	{
		std::vector<std::string> list = genicam_camera_->GetAvailableEnumEntriesAsSymbolics("AcquisitionMode");
		SetAllowedValues("CAM AcquisitionMode", list);
	}
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::On_CAM_AcquisitionMaxFrameRate(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	std::string str_value;
	reinterpret_cast<MM::Property*>(pProp)->SetReadOnly(!genicam_camera_->IsNodeWritable("AcquisitionMaxFrameRate"));
	
	if (pAct == MM::BeforeGet)
	{
//		if(!genicam_camera_->IsNodeReadable("AcquisitionMaxFrameRate"))
//		{
//			return ERR_NODE_NOT_READABLE;
//		}
		pProp->Set("0");
	}
	else if (pAct == MM::AfterSet)
	{
		if (!genicam_camera_->IsNodeWritable("AcquisitionMaxFrameRate"))
		{
			return ERR_NODE_NOT_WRITABLE;
		}
		pProp->Get(str_value);
		if (strcmp(str_value.c_str(), "1") == 0)
			genicam_camera_->ExecuteCommand("AcquisitionMaxFrameRate");
	}
	return DEVICE_OK;
}




int CmE5AQ8CXP6D_CameraDevice::OnBinning(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	long bin = 1;

   if (pAct == MM::BeforeGet)
   {
      pProp->Set(bin);
   }
   else if (pAct == MM::AfterSet)
   {
	   pProp->Get(bin);
   }
   return DEVICE_OK;
}



int CmE5AQ8CXP6D_CameraDevice::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	// eliminate warnings
	numImages;
	interval_ms;
	stopOnOverflow;

	// eliminate warnings
	interval_ms;

	int fg_format;
	int siso_result = Fg_getParameter(framegrabber_, FG_FORMAT, &fg_format, selected_camera_.PortIndex);
	switch (fg_format)
	{
	case FG_GRAY:
	case FG_GRAY10:
	case FG_GRAY12:
	case FG_GRAY14:
	case FG_GRAY16:
	case FG_GRAY32:
		break;
	default:
		stringstream ss;
		string s;
		FG_GetEnumNameFromEnumValue(FG_FORMAT, fg_format, s);
		ss << "StartSequenceAcquisition()" << endl << "FG_FORMAT (\"" << s << "\") not yet implemented.";
		SetErrorText(DEVICE_NOT_YET_IMPLEMENTED, ss.str().c_str());
		return DEVICE_NOT_YET_IMPLEMENTED;
	}

	const unsigned int c_MemoryBuffers(16);
	dma_mem_ = Fg_AllocMemEx(framegrabber_, GetImageBufferSize() * c_MemoryBuffers, c_MemoryBuffers);
	if (dma_mem_ == NULL)
		return DEVICE_OUT_OF_MEMORY;

	apcdata_.pThis = this;
	apcdata_.fg = framegrabber_;
	apcdata_.mem = dma_mem_;
	apcdata_.displayid = 0;
	apcdata_.port = selected_camera_.PortIndex;

	apc_ctrl_.data = &apcdata_;
	apc_ctrl_.version = 0;
	apc_ctrl_.flags = FG_APC_DEFAULTS;
	apc_ctrl_.func = ApcFunc;
	apc_ctrl_.timeout = 5;
	siso_result = Fg_registerApcHandler(framegrabber_, selected_camera_.PortIndex, &apc_ctrl_, FG_APC_CONTROL_BASIC);

	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK) {
		return ret;
	}
	/*int sgc_result =*/ Sgc_executeCommand(sgc_camera_handle_, "AcquisitionStart");
	siso_result = Fg_AcquireEx(framegrabber_, selected_camera_.PortIndex, numImages, ACQ_STANDARD, dma_mem_);

	is_sequence_acquisition_ = true;
	return DEVICE_OK;


//	SetErrorText(DEVICE_NOT_YET_IMPLEMENTED, "StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) has not yet been implemented.");
//	return DEVICE_NOT_YET_IMPLEMENTED;
}

int CmE5AQ8CXP6D_CameraDevice::StartSequenceAcquisition(double interval_ms)
{
	// eliminate warnings
	interval_ms;

	int fg_format;
	int siso_result = Fg_getParameter(framegrabber_, FG_FORMAT, &fg_format, selected_camera_.PortIndex);
	switch (fg_format)
	{
	case FG_GRAY:
	case FG_GRAY10:
	case FG_GRAY12:
	case FG_GRAY14:
	case FG_GRAY16:
	case FG_GRAY32:
		break;
	default:
		stringstream ss;
		string s;
		FG_GetEnumNameFromEnumValue(FG_FORMAT, fg_format, s);
		ss << "StartSequenceAcquisition()" << endl << "FG_FORMAT (\"" << s << "\") not yet implemented.";
		SetErrorText(DEVICE_NOT_YET_IMPLEMENTED, ss.str().c_str());
		return DEVICE_NOT_YET_IMPLEMENTED;
	}

	const unsigned int c_MemoryBuffers(16);
	dma_mem_ = Fg_AllocMemEx(framegrabber_, GetImageBufferSize() * c_MemoryBuffers, c_MemoryBuffers);
	if (dma_mem_ == NULL)
		return DEVICE_OUT_OF_MEMORY;

	apcdata_.pThis = this;
	apcdata_.fg = framegrabber_;
	apcdata_.mem = dma_mem_;
	apcdata_.displayid = 0;
	apcdata_.port = selected_camera_.PortIndex;

	apc_ctrl_.data = &apcdata_;
	apc_ctrl_.version = 0;
	apc_ctrl_.flags = FG_APC_DEFAULTS;
	apc_ctrl_.func = ApcFunc;
	apc_ctrl_.timeout = 5;
	siso_result = Fg_registerApcHandler(framegrabber_, selected_camera_.PortIndex, &apc_ctrl_, FG_APC_CONTROL_BASIC);

	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK) {
		return ret;
	}
	/*int sgc_result =*/ Sgc_executeCommand(sgc_camera_handle_, "AcquisitionStart");
	siso_result = Fg_AcquireEx(framegrabber_, selected_camera_.PortIndex, GRAB_INFINITE, ACQ_STANDARD, dma_mem_);

	is_sequence_acquisition_ = true;
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::StopSequenceAcquisition()
{
	Fg_registerApcHandler(framegrabber_, selected_camera_.PortIndex, NULL, FG_APC_CONTROL_BASIC);
	/*int sgc_result =*/ Sgc_executeCommand(sgc_camera_handle_, "AcquisitionStop");
	int siso_result = Fg_stopAcquireEx(framegrabber_, selected_camera_.PortIndex, dma_mem_, 0);
	siso_result = Fg_FreeMemEx(framegrabber_, dma_mem_);
	dma_mem_ = NULL;
	is_sequence_acquisition_ = false;
	return DEVICE_OK;
}

int CmE5AQ8CXP6D_CameraDevice::PrepareSequenceAcqusition()
{
	// Nothing to do.
	return DEVICE_OK;
}

bool CmE5AQ8CXP6D_CameraDevice::IsCapturing()
{
	return is_sequence_acquisition_;
}


//####################################################################


