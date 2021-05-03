///////////////////////////////////////////////////////////////////////////////
// FILE:          SiSoGenICamCameraPort.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   GenICam IPort implementation of CXP camera connected through
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

#include "SiSoGenICamCameraPort.h"


CSiSoGenICamCameraPort::CSiSoGenICamCameraPort(SgcCameraHandle* sgc_camera_handle) :
	m_SgcCameraHandle(sgc_camera_handle)
{
}

CSiSoGenICamCameraPort::~CSiSoGenICamCameraPort(void)
{
}

//! Reads a chunk of bytes from the port
void CSiSoGenICamCameraPort::Read(void *pBuffer, int64_t Address, int64_t Length)
{
	Sgc_memoryReadFromCamera(m_SgcCameraHandle, pBuffer, Address, Length);
}

//! Writes a chunk of bytes to the port
void CSiSoGenICamCameraPort::Write(const void *pBuffer, int64_t Address, int64_t Length)
{
	Sgc_memoryWriteToCamera(m_SgcCameraHandle, pBuffer, Address, Length);
}


//! Always assume R/W access to the camera
GenApi::EAccessMode CSiSoGenICamCameraPort::GetAccessMode() const
{
	using namespace GenApi;
	return RW;
}