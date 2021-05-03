///////////////////////////////////////////////////////////////////////////////
// FILE:          SiSoGenICamCamera.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   GenICam implementation of CXP camera connected through
//                mE5AQ8-CXP6D framegrabber board
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



#pragma once

#include "SiSoGenICamCameraPort.h"
#include "Basler_mE5AQ8-CXP6D_CameraDevice.h"

#include <stdlib.h>

#include <siso_genicam.h>

//#define GENICAM_USER_ALWAYS_LINK_RELEASE 
#include <GenICam.h>

// Don't use namespaces in header files!
//using namespace GenApi;
class CmE5AQ8CXP6D_CameraDevice;
class CSiSoGenICamCamera
{
public:
	CSiSoGenICamCamera(SgcCameraHandle* sgc_camera_handle, CmE5AQ8CXP6D_CameraDevice* parent);
//	CSiSoGenICamCamera(SgcCameraHandle* sgc_camera_handle);
	~CSiSoGenICamCamera(void);

	GenApi::INodeMap* GetNodeMap();

	bool IsNodeAvailable(std::string s);
	bool IsNodeAvailable(const char *s);
	bool IsNodeAvailable(GenApi::INode *ptrNode);

	bool IsNodeReadable(std::string s);
	bool IsNodeReadable(const char *s);
	bool IsNodeReadable(GenApi::INode *ptrNode);

	bool IsNodeWritable(std::string s);
	bool IsNodeWritable(const char *s);
	bool IsNodeWritable(GenApi::INode *ptrNode);
	void AddToLog(std::string msg);

	std::vector<std::string> GetAvailableEnumEntriesAsSymbolics(const char*enumName);
	std::vector<std::string> GetAvailableEnumEntriesAsSymbolics(GenApi::CEnumerationPtr ptrEnumeration);

	std::vector<GenApi::CEnumEntryPtr> GetAvailableEnumEntries(const char*enumName);
	std::vector<GenApi::CEnumEntryPtr> GetAvailableEnumEntries(GenApi::CEnumerationPtr ptrEnumeration);

	std::string GetValueAsString(const char *nodeName);
	void SetValueFromString(const char *nodeName, const char *strValue);
	void SetValueFromString(const char *nodeName, std::string &strValue);

	int64_t GetIntegerValue(const char *nodeName);
	void SetIntegerValue(const char *nodeName, int64_t intValue);

	double GetFloatValue(const char *nodeName);
	void SetFloatValue(const char *nodeName, double floatValue);

	bool GetBooleanValue(const char *nodeName);
	void SetBooleanValue(const char *nodeName, bool booleanValue);

	void ExecuteCommand(const char *nodeName);
	bool IsCommandDone(const char *nodeName);

private:
	CSiSoGenICamCameraPort* m_pPort;
	SgcCameraHandle* m_CameraHandle;
	CmE5AQ8CXP6D_CameraDevice* m_pParent;
	size_t m_XMLBufferSize;
	char* m_XMLBuffer;
	GenApi::CNodeMapFactory* m_pNodeMapFactory;
	GenApi::INodeMap* m_pNodeMap;
};

