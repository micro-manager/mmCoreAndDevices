///////////////////////////////////////////////////////////////////////////////
// FILE:          SiSoGenICamCamera.cpp
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

#include "SiSoGenICamCamera.h"
#include <GenICam.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

using namespace GenApi;
using namespace std;

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

CSiSoGenICamCamera::CSiSoGenICamCamera(SgcCameraHandle* sgc_camera_handle, CmE5AQ8CXP6D_CameraDevice* p_parent) :
	m_pPort(NULL),
	m_CameraHandle(sgc_camera_handle),
	m_pParent(p_parent),
	m_XMLBufferSize(0),
	m_XMLBuffer(NULL),
	m_pNodeMapFactory(NULL),
	m_pNodeMap(NULL)
{
	try
	{
		AddToLog("SiSoGenICamCamera() called.\n");
		m_pPort = new CSiSoGenICamCameraPort(m_CameraHandle);
#ifndef USE_XML_FROM_FILE
		int siso_res = Sgc_getGenICamXML(m_CameraHandle, m_XMLBuffer, &m_XMLBufferSize);
		char s[255];
		sprintf(s, "   Sgc_getGenICamXML() returned: %s\n", Sgc_getErrorDescription(siso_res)); 
		std::string str(s);
		AddToLog(str);
		if (siso_res != SGC_OK)
		{
			throw GenericException("Failed to download XML file from camera.", __FILE__, __LINE__);
		}
		m_XMLBuffer = (char*)malloc(m_XMLBufferSize);
		siso_res = Sgc_getGenICamXML(m_CameraHandle, m_XMLBuffer, &m_XMLBufferSize);
		sprintf(s, "   Sgc_getGenICamXML() returned: %s\n", Sgc_getErrorDescription(siso_res)); 
		str.assign(s);
		AddToLog(str);
		if (siso_res != SGC_OK)
		{
			free(m_XMLBuffer);
			throw GenericException("Failed to download XML file from camera.", __FILE__, __LINE__);
		}
#else
		int m_XMLBufferSize = GetFileSize("camera.xml");
		m_XMLBuffer = (char *)malloc(m_XMLBufferSize);
		FILE *fp = fopen("camera.xml", "rb");
		fread(m_XMLBuffer, m_XMLBufferSize, 1, fp);
		fclose(fp);
#endif
		bool isZippedXml = (_strnicmp( reinterpret_cast<const char*>(&m_XMLBuffer[0]), "PK\x03\x04", 4 ) == 0);

		if (isZippedXml)
		{
			m_pNodeMapFactory = new CNodeMapFactory(ContentType_ZippedXml, m_XMLBuffer, m_XMLBufferSize);
		}
		else
		{
			// Trimm XML (might have garbage bytes at the tail, which the XML parser doesn't like)
			int i = 0;
			while (m_XMLBuffer[m_XMLBufferSize - 1 - i] != '>')
				i++;
			m_XMLBufferSize = m_XMLBufferSize - i;

			m_pNodeMapFactory = new CNodeMapFactory(ContentType_Xml, m_XMLBuffer, m_XMLBufferSize);
		}
		m_pNodeMap = m_pNodeMapFactory->CreateNodeMap();
		m_pNodeMap->Connect(m_pPort);
		free(m_XMLBuffer);
	}
	catch (const GenericException &e)
	{
		// Error handling.
		AddToLog(e.GetDescription());
		cerr << "An exception occurred." << endl
			<< e.GetDescription() << endl;
		throw e;
	}
}

void CSiSoGenICamCamera::AddToLog(std::string msg)
{
	if (m_pParent != NULL)
	{
		m_pParent->AddToLog(msg);
	}
}

CSiSoGenICamCamera::~CSiSoGenICamCamera(void)
{
//	delete m_pNodeMap;
	delete m_pNodeMapFactory;
//	delete m_XMLBuffer;
	delete m_pPort;
}

GenApi::INodeMap* CSiSoGenICamCamera::GetNodeMap()
{
	return m_pNodeMap;
}


bool CSiSoGenICamCamera::IsNodeAvailable(string s)
{
	INode *ptr_node = GetNodeMap()->GetNode(gcstring(s.c_str()));
	bool is_available = IsAvailable(ptr_node);
	return is_available;
}

bool CSiSoGenICamCamera::IsNodeAvailable(const char *s)
{
	INode *ptr_node = GetNodeMap()->GetNode(s);
	bool is_available = IsAvailable(ptr_node);
	return is_available;
}

bool CSiSoGenICamCamera::IsNodeAvailable(INode *ptrNode)
{
	bool is_available = IsAvailable(ptrNode);
	return is_available;
}


bool CSiSoGenICamCamera::IsNodeReadable(string s)
{
	INode *ptr_node = GetNodeMap()->GetNode(gcstring(s.c_str()));
	bool is_readable = IsReadable(ptr_node);
	return is_readable;
}

bool CSiSoGenICamCamera::IsNodeReadable(const char *s)
{
	INode *ptr_node = GetNodeMap()->GetNode(s);
	bool is_readable = IsReadable(ptr_node);
	return is_readable;
}

bool CSiSoGenICamCamera::IsNodeReadable(INode *ptrNode)
{
	bool is_readable = IsReadable(ptrNode);
	return is_readable;
}


bool CSiSoGenICamCamera::IsNodeWritable(string s)
{
	INode *ptr_node = GetNodeMap()->GetNode(gcstring(s.c_str()));
	bool is_writable = IsWritable(ptr_node);
	return is_writable;
}

bool CSiSoGenICamCamera::IsNodeWritable(const char *s)
{
	INode *ptr_node = GetNodeMap()->GetNode(s);
	bool is_writable = IsWritable(ptr_node);
	return is_writable;
}

bool CSiSoGenICamCamera::IsNodeWritable(INode *ptrNode)
{
	bool is_writable = IsWritable(ptrNode);
	return is_writable;
}


vector<string> CSiSoGenICamCamera::GetAvailableEnumEntriesAsSymbolics(const char*enumName)
{
	vector<string> list;
	INode *p_node = m_pNodeMap->GetNode(enumName);
	if (IsAvailable(p_node) && (p_node->GetPrincipalInterfaceType() == intfIEnumeration))
		return GetAvailableEnumEntriesAsSymbolics(CEnumerationPtr(p_node));
	return list;
}

vector<string> CSiSoGenICamCamera::GetAvailableEnumEntriesAsSymbolics(CEnumerationPtr ptrEnumeration)
{
	NodeList_t nodelist;
	vector<string> list;
	if (IsAvailable(ptrEnumeration))
	{
		ptrEnumeration->GetEntries(nodelist);
		for (NodeList_t::iterator it = nodelist.begin(); it != nodelist.end(); it++)
		{
			if (IsAvailable(*it))
				list.push_back(string((*it)->GetDisplayName().c_str()));
		}
	}
	return list;
}

vector<CEnumEntryPtr> CSiSoGenICamCamera::GetAvailableEnumEntries(const char*enumName)
{
	vector<CEnumEntryPtr> list;
	INode *p_node = m_pNodeMap->GetNode(enumName);
	if (IsAvailable(p_node) && (p_node->GetPrincipalInterfaceType() == intfIEnumeration))
		return GetAvailableEnumEntries(CEnumerationPtr(p_node));
	return list;
}

vector<CEnumEntryPtr> CSiSoGenICamCamera::GetAvailableEnumEntries(CEnumerationPtr ptrEnumeration)
{
	NodeList_t nodelist;
	vector<CEnumEntryPtr> list;
	if (IsAvailable(ptrEnumeration))
	{
		ptrEnumeration->GetEntries(nodelist);
		for (NodeList_t::iterator it = nodelist.begin(); it != nodelist.end(); it++)
		{
			if (IsAvailable(*it))
				list.push_back(CEnumEntryPtr(*it));
		}
	}
	return list;
}

string CSiSoGenICamCamera::GetValueAsString(const char *nodeName)
{
	IValue *p_value = dynamic_cast<IValue *>(m_pNodeMap->GetNode(nodeName));
	if (IsReadable(p_value))
	{
		gcstring s = p_value->ToString();
		return string(s.c_str());
	}
	return NULL;
}

void CSiSoGenICamCamera::SetValueFromString(const char *nodeName, const char *strValue)
{
	IValue *p_value = dynamic_cast<IValue *>(m_pNodeMap->GetNode(nodeName));
	if (IsWritable(p_value))
	{
		try
		{
			p_value->FromString(gcstring(strValue));
		}
		catch (GenericException e)
		{
			throw e;
		}
	}
}

void CSiSoGenICamCamera::SetValueFromString(const char *nodeName, string &strValue)
{
	INode *p_node = m_pNodeMap->GetNode(nodeName);
	if (p_node == NULL)
	{
		throw GenericException("Could not get node.", __FILE__, __LINE__, "SetValueFromString()", nodeName, "GenericException");
	}
	IValue *p_value = dynamic_cast<IValue *>(p_node);
	if (IsWritable(p_value))
	{
		p_value->FromString(gcstring(strValue.c_str()));

	}
}


int64_t CSiSoGenICamCamera::GetIntegerValue(const char *nodeName)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "GetIntegerValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIInteger)
		{
			throw GenericException("Node is not of Type IInteger.", __FILE__, __LINE__, "GetIntegerValue()", nodeName, "GenericException");
		}
		CIntegerPtr ptrInteger(p_node);
		if (IsReadable(ptrInteger))
		{
			return ptrInteger->GetValue();
		}
		else
		{
			throw GenericException("Node is not Readable.", __FILE__, __LINE__, "GetIntegerValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

void CSiSoGenICamCamera::SetIntegerValue(const char *nodeName, int64_t intValue)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "SetIntegerValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIInteger)
		{
			throw GenericException("Node is not of Type IInteger.", __FILE__, __LINE__, "SetIntegerValue()", nodeName, "GenericException");
		}
		CIntegerPtr ptrInteger(p_node);
		if (IsWritable(ptrInteger))
		{
			ptrInteger->SetValue(intValue);
		}
		else
		{
			throw GenericException("Node is not Writable.", __FILE__, __LINE__, "SetIntegerValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

double CSiSoGenICamCamera::GetFloatValue(const char *nodeName)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "GetFloatValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIFloat)
		{
			throw GenericException("Node is not of Type IFloat.", __FILE__, __LINE__, "GetFloatValue()", nodeName, "GenericException");
		}
		CFloatPtr ptrFloat(p_node);
		if (IsReadable(ptrFloat))
		{
			return ptrFloat->GetValue();
		}
		else
		{
			throw GenericException("Node is not Readable.", __FILE__, __LINE__, "GetFloatValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

void CSiSoGenICamCamera::SetFloatValue(const char *nodeName, double floatValue)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "SetFloatValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIFloat)
		{
			throw GenericException("Node is not of Type IInteger.", __FILE__, __LINE__, "SetFloatValue()", nodeName, "GenericException");
		}
		CFloatPtr ptrFloat(p_node);
		if (IsWritable(ptrFloat))
		{
			ptrFloat->SetValue(floatValue);
		}
		else
		{
			throw GenericException("Node is not Writable.", __FILE__, __LINE__, "SetFloatValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

bool CSiSoGenICamCamera::GetBooleanValue(const char *nodeName)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "GetBooleanValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIBoolean)
		{
			throw GenericException("Node is not of Type IBoolean.", __FILE__, __LINE__, "GetBooleanValue()", nodeName, "GenericException");
		}
		CBooleanPtr ptrBoolean(p_node);
		if (IsReadable(ptrBoolean))
		{
			return ptrBoolean->GetValue();
		}
		else
		{
			throw GenericException("Node is not Readable.", __FILE__, __LINE__, "GetBooleanValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

void CSiSoGenICamCamera::SetBooleanValue(const char *nodeName, bool booleanValue)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "SetBooleanValue()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfIBoolean)
		{
			throw GenericException("Node is not of Type IBoolean.", __FILE__, __LINE__, "SetBooleanValue()", nodeName, "GenericException");
		}
		CBooleanPtr ptrBoolean(p_node);
		if (IsWritable(ptrBoolean))
		{
			ptrBoolean->SetValue(booleanValue);
		}
		else
		{
			throw GenericException("Node is not Writable.", __FILE__, __LINE__, "SetBooleanValue()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

void CSiSoGenICamCamera::ExecuteCommand(const char *nodeName)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "ExecuteCommand()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfICommand)
		{
			throw GenericException("Node is not of Type ICommand.", __FILE__, __LINE__, "ExecuteCommand()", nodeName, "GenericException");
		}
		CCommandPtr ptrCommand(p_node);
		if (IsWritable(ptrCommand))
		{
			ptrCommand->Execute();
		}
		else
		{
			throw GenericException("Node is not Writable.", __FILE__, __LINE__, "ExecuteCommand()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}

bool CSiSoGenICamCamera::IsCommandDone(const char *nodeName)
{
	try
	{
		INode *p_node = m_pNodeMap->GetNode(nodeName);
		if (p_node == NULL)
		{
			throw GenericException("Could not get Node.", __FILE__, __LINE__, "IsCommandDone()", nodeName, "GenericException");
		}
		if (p_node->GetPrincipalInterfaceType() != intfICommand)
		{
			throw GenericException("Node is not of Type ICommand.", __FILE__, __LINE__, "IsCommandDone()", nodeName, "GenericException");
		}
		CCommandPtr ptrCommand(p_node);
		if (IsReadable(ptrCommand))
		{
			return ptrCommand->IsDone();
		}
		else
		{
			throw GenericException("Node is not Readable.", __FILE__, __LINE__, "IsCommandDone()", nodeName, "GenericException");
		}
	}
	catch (GenericException e)
	{
		throw e;
	}
}



#if 0
std::vector<std::string> CSiSoGenICamCamera::GetExposureModeList()
{
	StringList_t symbolics;
	vector<string> string_list;
	CEnumerationPtr(m_pNodeMap->GetNode("ExposureMode"))->GetSymbolics(symbolics);
	for (StringList_t::iterator it = symbolics.begin(); it != symbolics.end(); it++)
	{
		string_list.push_back((*it).c_str());
	}
	return string_list;
}
#endif
