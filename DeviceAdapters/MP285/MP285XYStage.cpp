//////////////////////////////////////////////////////////////////////////////
// FILE:          MP285XYStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   MP285s Controller Driver
//
// COPYRIGHT:     Sutter Instrument,
//				  Mission Bay Imaging, San Francisco, 2011
//                All rights reserved
//
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation.
//                
//                You should have received a copy of the GNU Lesser General Public
//                License along with the source distribution; if not, write to
//                the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
//                Boston, MA  02111-1307  USA
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
//
// AUTHOR:        Lon Chu (lonchu@yahoo.com), created on June 2011
//

#ifdef WIN32
   #include <windows.h>
#endif
#include "FixSnprintf.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "MP285Error.h"
#include "MP285XYStage.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////////
// XYStage methods required by the API
///////////////////////////////////////////////////////////////////////////////
//
// XYStage - two axis stage device.
// Note that this adapter uses two coordinate systems.  There is the adapters own coordinate
// system with the X and Y axis going the 'Micro-Manager standard' direction
// Then, there is the MP285s native system.  All functions using 'steps' use the MP285 system
// All functions using Um use the Micro-Manager coordinate system
//

//
// XY Stage constructor
//
XYStage::XYStage() :
    m_yInitialized(false)
    //range_measured_(false),
    //m_nAnswerTimeoutMs(1000)
    //stepSizeUm_(1.0),
    //set speed & accel variables?
    //originX_(0),
    //originY_(0)
{
    InitializeDefaultErrorMessages();

    // create pre-initialization properties
    // ------------------------------------
    // NOTE: pre-initialization properties contain parameters which must be defined fo
    // proper startup

    m_nAnswerTimeoutMs = MP285::Instance()->GetTimeoutInterval();
    m_nAnswerTimeoutTrys = MP285::Instance()->GetTimeoutTrys();

    // Name, read-only (RO)
    char sXYName[120];
	sprintf(sXYName, "%s%s", MP285::Instance()->GetMPStr(MP285::MPSTR_XYDevNameLabel).c_str(), MM::g_Keyword_Name);
    int ret = CreateProperty(sXYName, MP285::Instance()->GetMPStr(MP285::MPSTR_XYStgaeDevName).c_str(), MM::String, true);

    std::ostringstream osMessage;

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::class-constructor> CreateProperty(" << sXYName << "=" << MP285::Instance()->GetMPStr(MP285::MPSTR_XYStgaeDevName).c_str() << "), ReturnCode=" << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    // Description, RO
    char sXYDesc[120];
	sprintf(sXYDesc, "%s%s", MP285::Instance()->GetMPStr(MP285::MPSTR_XYDevDescLabel).c_str(), MM::g_Keyword_Description);
    ret = CreateProperty(sXYDesc, "MP-285 XY Stage Driver", MM::String, true);

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::class-constructor> CreateProperty(" << sXYDesc << " = MP-285 XY Stage Driver), ReturnCode=" << ret;
		this->LogMessage(osMessage.str().c_str());
	}
}

//
// XY Stage destructor
//
XYStage::~XYStage()
{
    Shutdown();
}

void XYStage::GetName(char* sName) const
{
    CDeviceUtils::CopyLimitedString(sName, MP285::Instance()->GetMPStr(MP285::MPSTR_XYStgaeDevName).c_str());
}

//
// Performs device initialization.
// Additional properties can be defined here too.
//
int XYStage::Initialize()
{
    std::ostringstream osMessage;

    if (!MP285::Instance()->GetDeviceAvailability()) return DEVICE_NOT_CONNECTED;

    // int ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str(), "undefined", MM::String, true);  // Get Position X 
	CPropertyAction* pActOnGetPosX = new CPropertyAction(this, &XYStage::OnGetPositionX);
	char sPosX[20];
	double dPosX = MP285::Instance()->GetPositionX();
	sprintf(sPosX, "%ld", (long)(dPosX * (double)MP285::Instance()->GetUm2UStep()));
	int ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str(), sPosX, MM::Integer, false, pActOnGetPosX);  // Get Position X 

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::Initialize> CreateProperty(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str() << " = " << sPosX << "), ReturnCode = " << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK)  return ret;

    //ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str(), "undefined", MM::String, true);  // Get Position Y 
	CPropertyAction* pActOnGetPosY = new CPropertyAction(this, &XYStage::OnGetPositionY);
	char sPosY[20];
    double dPosY = MP285::Instance()->GetPositionY();
	sprintf(sPosY, "%ld", (long)(dPosY * (double)MP285::Instance()->GetUm2UStep()));
	ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str(), sPosY, MM::Integer, false, pActOnGetPosY);  // Get Position Y 
 
    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::Initialize> CreateProperty(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str() << " = " << sPosY << "), ReturnCode = " << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK)  return ret;

    ret = GetPositionUm(dPosX, dPosY);

	sprintf(sPosX, "%ld", (long)(dPosX*(double)MP285::Instance()->GetUm2UStep()));
	sprintf(sPosY, "%ld", (long)(dPosY*(double)MP285::Instance()->GetUm2UStep()));

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::Initialize> GetPosSteps(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str() << " = " << sPosX << ",";
		osMessage << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str() << " = " << sPosY << "), ReturnCode = " << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret!=DEVICE_OK) return ret;

    CPropertyAction* pActOnSetPosX = new CPropertyAction(this, &XYStage::OnSetPositionX);
	sprintf(sPosX, "%.2f", dPosX);
    ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionX).c_str(), sPosX, MM::Float, false, pActOnSetPosX);  // Set Position X 
    //ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionX).c_str(), "Undefined", MM::Integer, true);  // Set Position X 

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::Initialize> CreateProperty(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionX).c_str() << sPosX << "), ReturnCode = " << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK)  return ret;

    CPropertyAction* pActOnSetPosY = new CPropertyAction(this, &XYStage::OnSetPositionY);
	sprintf(sPosY, "%.2f", dPosY);
    ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionY).c_str(), sPosY, MM::Float, false, pActOnSetPosY);  // Set Position Y 
    //ret = CreateProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionY).c_str(), "Undefined", MM::Integer, true);  // Set Position Y 

    if (MP285::Instance()->GetDebugLogFlag() > 0)
    {
		osMessage.str("");
		osMessage << "<XYStage::Initialize> CreateProperty(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionY).c_str() << sPosY << "), ReturnCode = " << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK)  return ret;


    ret = UpdateStatus();
    if (ret != DEVICE_OK) return ret;

    m_yInitialized = true;
    return DEVICE_OK;
}

//
// shutdown X-Y stage
//
int XYStage::Shutdown()
{
    m_yInitialized= false;
    MP285::Instance()->SetDeviceAvailable(false);
    return DEVICE_OK;
}

//
// Set Motion Mode (1: relatice, 0: absolute)
//
int XYStage::SetMotionMode(long lMotionMode)
{
    std::ostringstream osMessage;
    unsigned char sCommand[6] = { 0x00, MP285::MP285_TxTerm, 0x0A, 0x00, 0x00, 0x00 };
    unsigned char sResponse[64];
    int ret = DEVICE_OK;
        
    if (lMotionMode == 0)
        sCommand[0] = 'a';
    else
        sCommand[0] = 'b';

    ret = WriteCommand(sCommand, 3);

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		osMessage.str("");
		osMessage << "<XYStage::SetMotionMode> = [" << lMotionMode << "," << sCommand[0] << "], Returncode =" << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK) return ret;

    ret = ReadMessage(sResponse, 2);

    if (ret != DEVICE_OK) return ret;

    MP285::Instance()->SetMotionMode(lMotionMode);

    return DEVICE_OK;
}

//
// Returns current X-Y position in �m.
//
int XYStage::GetPositionUm(double& dXPosUm, double& dYPosUm)
{
    long lXPosSteps = 0;
    long lYPosSteps = 0;

    int ret = GetPositionSteps(lXPosSteps, lYPosSteps);

    if (ret != DEVICE_OK) return ret;

    dXPosUm = (double)lXPosSteps / (double)MP285::Instance()->GetUm2UStep();
    dYPosUm = (double)lYPosSteps / (double)MP285::Instance()->GetUm2UStep();

    ostringstream osMessage;

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		osMessage.str("");
		osMessage << "<MP285::XYStage::GetPositionUm> (x=" << dXPosUm << ", y=" << dYPosUm << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    MP285::Instance()->SetPositionX(dXPosUm);
    MP285::Instance()->SetPositionY(dYPosUm);

    //char sPosition[20];
    //sprintf(sPosition, "%ld", lXPosSteps);
    //ret = SetProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str(), sPosition);

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		osMessage.str("");
		osMessage << "<XYStage::GetPositionUm> X=[" << dXPosUm << /*"," << sPosition <<*/ "], Returncode=" << ret ;
		this->LogMessage(osMessage.str().c_str());
	}

    //if (ret != DEVICE_OK) return ret;

    //sprintf(sPosition, "%ld", lYPosSteps);
    //ret = SetProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str(), sPosition);

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		osMessage.str("");
		osMessage << "<XYStage::GetPositionUm> Y=[" << dYPosUm << /*"," << sPosition <<*/ "], Returncode=" << ret ;
		this->LogMessage(osMessage.str().c_str());
	}

    //if (ret != DEVICE_OK) return ret;

    //ret = UpdateStatus();
    //if (ret != DEVICE_OK) return ret;


    return DEVICE_OK;
}

//
// Move x-y stage to a relative distance from current position in �m
//
int XYStage::SetRelativePositionUm(double dXPosUm, double dYPosUm)
{
	int ret = DEVICE_OK;
    ostringstream osMessage;

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		//osMessage.str("");
		osMessage << "<XYStage::SetRelativePositionUm> (x=" << dXPosUm << ", y=" << dYPosUm << ")";
		this->LogMessage(osMessage.str().c_str());
	}

	// set relative motion mode
	if (MP285::Instance()->GetMotionMode() == 0)
	{
		ret = SetMotionMode(1);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage.str("");
			osMessage << "<XYZStage::SetRelativePositionUm> (" << MP285::Instance()->GetMPStr(MP285::MPSTR_MotionMode).c_str() << " = <RELATIVE>), ReturnCode = " << ret;
			this->LogMessage(osMessage.str().c_str());
		}

	    if (ret != DEVICE_OK) return ret;
	}


    // convert um to steps 
    long lXPosSteps = (long)(dXPosUm * (double)MP285::Instance()->GetUm2UStep());
    long lYPosSteps = (long)(dYPosUm * (double)MP285::Instance()->GetUm2UStep());

    // send move command to controller
	// set the Relative motion mode
	ret = _SetPositionSteps(lXPosSteps, lYPosSteps, 0L);

    if (ret != DEVICE_OK) return ret;

    double dPosX = 0.;
    double dPosY = 0.;

    ret = GetPositionUm(dPosX, dPosY);

    if (ret != DEVICE_OK) return ret;

    return ret;
}

//
// Move 2 x-y position in �m
//
int XYStage::SetPositionUm(double dXPosUm, double dYPosUm)
{
	int ret = DEVICE_OK;
    ostringstream osMessage;

    if (MP285::Instance()->GetDebugLogFlag() > 1)
    {
		//osMessage.str("");
		osMessage << "<XYStage::SetPositionUm> (x=" << dXPosUm << ", y=" << dYPosUm << ")";
		this->LogMessage(osMessage.str().c_str());
	}

	// set absolute motion mode
	if (MP285::Instance()->GetMotionMode() != 0)
	{
		ret = SetMotionMode(0);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage.str("");
			osMessage << "<XYStage::SetPositionUm> (" << MP285::Instance()->GetMPStr(MP285::MPSTR_MotionMode).c_str() << " = <ABSOLUTE>), ReturnCode = " << ret;
			this->LogMessage(osMessage.str().c_str());
		}
		
		if (ret != DEVICE_OK) return ret;
	}

    // convert um to steps 
    long lXPosSteps = (long)(dXPosUm * (double)MP285::Instance()->GetUm2UStep());
    long lYPosSteps = (long)(dYPosUm * (double)MP285::Instance()->GetUm2UStep());
	long lZPosSteps = (long)MP285::Instance()->GetPositionZ() * (long)MP285::Instance()->GetUm2UStep();

    // send move command to controller
    ret = _SetPositionSteps(lXPosSteps, lYPosSteps, lZPosSteps);

    if (ret != DEVICE_OK) return ret;

    double dPosX = 0.;
    double dPosY = 0.;

    ret = GetPositionUm(dPosX, dPosY);

    if (ret != DEVICE_OK) return ret;

    return ret;
}
  
//
// Returns current position in steps.
//
int XYStage::GetPositionSteps(long& lXPosSteps, long& lYPosSteps)
{
    // get current position
    unsigned char sCommand[6] = { 0x63, MP285::MP285_TxTerm, 0x0A, 0x00, 0x00, 0x00 };
    int ret = WriteCommand(sCommand, 3);

    if (ret != DEVICE_OK)  return ret;

    unsigned char sResponse[64];
    memset(sResponse, 0, 64);

    bool yCommError = false;
    int nTrys = 0;

    while (!yCommError && nTrys < MP285::Instance()->GetTimeoutTrys())
    {
        long lZPosSteps = (long) (MP285::Instance()->GetPositionZ() * (double)MP285::Instance()->GetUm2UStep());

        ret = ReadMessage(sResponse, 14);

        ostringstream osMessage;
        char sCommStat[30];
        int nError = CheckError(sResponse[0]);
        yCommError = (sResponse[0] == 0) ? false : nError != 0;
        if (yCommError)
        {
            if (nError == MPError::MPERR_SerialZeroReturn && nTrys < MP285::Instance()->GetTimeoutTrys()) { nTrys++; yCommError = false; }
			if (MP285::Instance()->GetDebugLogFlag() > 1)
			{
				osMessage.str("");
				osMessage << "<XYStage::GetPositionSteps> Response = (" << nError << "," << nTrys << ")" ;
			}
			sprintf(sCommStat, "Error Code ==> <%2x>", sResponse[0]);
        }
        else
        {
            lXPosSteps = *((long*)(&sResponse[0]));
            lYPosSteps = *((long*)(&sResponse[4]));
            lZPosSteps = *((long*)(&sResponse[8]));
            //MP285::Instance()->SetPositionX(lXPosSteps);
            //MP285::Instance()->SetPositionY(lYPosSteps);
            //MP285::Instance()->SetPositionZ(lZPosSteps);

			if (MP285::Instance()->GetDebugLogFlag() > 1)
			{
				osMessage.str("");
				osMessage << "<XYStage::GetPositionSteps> Response(X = <" << lXPosSteps << ">, Y = <" << lYPosSteps << ">, Z = <"<< lZPosSteps << ">), ReturnCode=" << ret;
			}
            nTrys = MP285::Instance()->GetTimeoutTrys();
            strcpy(sCommStat, "Success");
           
        }

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			this->LogMessage(osMessage.str().c_str());
		}

        //ret = SetProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_CommStateLabel).c_str(), sCommStat);
    }

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}

//
// Move x-y stage to a relative distance from current position in uSteps
//
int XYStage::SetRelativePositionSteps(long lXPosSteps, long lYPosSteps)
{
	int ret = DEVICE_OK;
    ostringstream osMessage;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::SetRelativePositionSteps> (x=" << lXPosSteps << ", y=" << lYPosSteps << ")";
		this->LogMessage(osMessage.str().c_str());
	}

	// set relative motion mode
	if (MP285::Instance()->GetMotionMode() == 0)
	{
		ret = SetMotionMode(1);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage.str("");
			osMessage << "<XYStage::SetRelativePositionSteps> (" << MP285::Instance()->GetMPStr(MP285::MPSTR_MotionMode).c_str() << " = <RELATIVE>), ReturnCode = " << ret;
			LogMessage(osMessage.str().c_str());
		}
		if (ret != DEVICE_OK) return ret;
	}

	ret = _SetPositionSteps(lXPosSteps, lYPosSteps, 0L);

	if (ret != DEVICE_OK) return ret;

	return DEVICE_OK;
}

//
// Move x-y stage to an absolute position in uSteps
//
int XYStage::SetPositionSteps(long lXPosSteps, long lYPosSteps)
{
	int ret = DEVICE_OK;
    ostringstream osMessage;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::SetPositionSteps> (x=" << lXPosSteps << ", y=" << lYPosSteps << ")";
		this->LogMessage(osMessage.str().c_str());
	}

	// set absolute motion mode
	if (MP285::Instance()->GetMotionMode() != 0)
	{
		ret = SetMotionMode(0);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage.str("");
			osMessage << "<XYStage::SetPositionSteps> (" << MP285::Instance()->GetMPStr(MP285::MPSTR_MotionMode).c_str() << " = <ABSOLUTE>), ReturnCode = " << ret;
			LogMessage(osMessage.str().c_str());
		}
		if (ret != DEVICE_OK) return ret;
	}

	long lZPosSteps = (long)MP285::Instance()->GetPositionZ() * (long)MP285::Instance()->GetUm2UStep();

	ret = _SetPositionSteps(lXPosSteps, lYPosSteps, lZPosSteps);

	if (ret != DEVICE_OK) return ret;

	return DEVICE_OK;
}

//
// Move x-y-z stage in uSteps
//
int XYStage::_SetPositionSteps(long lXPosSteps, long lYPosSteps, long lZPosSteps)
{
	int ret = DEVICE_OK;
    ostringstream osMessage;

    // get current position
    unsigned char sCommand[16];
    memset(sCommand, 0, 16);
    sCommand[0]  = 0x6D;
    sCommand[13] = MP285::MP285_TxTerm;
    sCommand[14] = 0x0A;
    long* plPositionX = (long*)(&sCommand[1]);
    *plPositionX = lXPosSteps;
    long* plPositionY = (long*)(&sCommand[5]);
    *plPositionY = lYPosSteps;
    long* plPositionZ = (long*)(&sCommand[9]);
    *plPositionZ = lZPosSteps;

    ret = WriteCommand(sCommand, 15);

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::_SetPositionSteps> Command(<0x6D>, X = <" << *plPositionX << ">,<" << *plPositionY << ">,<" << *plPositionZ << ">), ReturnCode=" << ret;
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK)  return ret;

	double dVelocity = (double)MP285::Instance()->GetVelocity() * (double)MP285::Instance()->GetUm2UStep();
	long lXSteps = 0;
	long lYSteps = 0;
	if (MP285::Instance()->GetMotionMode() == 0)
	{
		long lOldXPosSteps = (long)MP285::Instance()->GetPositionX();
		lXSteps = labs(lXPosSteps-lOldXPosSteps);
		long lOldYPosSteps = (long)MP285::Instance()->GetPositionY();
		lYSteps = labs(lYPosSteps-lOldYPosSteps);
	}
	else
	{
		lXSteps = labs(lXPosSteps);
		lYSteps = labs(lYPosSteps);
	}
    double dSec =  (lXSteps > lYSteps) ? (double)lXSteps / dVelocity : (double)lYSteps / dVelocity;
    long lSleep = (long)(dSec * 120.);
    CDeviceUtils::SleepMs(lSleep);
    
	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::_SetPositionSteps> Sleep..." << lSleep << " millisec...";
		this->LogMessage(osMessage.str().c_str());
	}

    bool yCommError = true;

    while (yCommError)
    {
        unsigned char sResponse[64];
        memset(sResponse, 0, 64);

        ret = ReadMessage(sResponse, 2);

        //char sCommStat[30];
		yCommError = CheckError(sResponse[0]) != MPError::MPERR_OK;
        //if (yCommError)
        //{
        //    sprintf(sCommStat, "Error Code ==> <%2x>", sResponse[0]);
        //}
        //else
        //{
        //    strcpy(sCommStat, "Success");
        //}

        //ret = SetProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_CommStateLabel).c_str(), sCommStat);
    }

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}

//
// stop and interrupt Z stage motion
//
int XYStage::Stop()
{
    unsigned char sCommand[6] = { 0x03, MP285::MP285_TxTerm , 0x00, 0x00, 0x00, 0x00};

    int ret = WriteCommand(sCommand, 2);

    ostringstream osMessage;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::Stop> (ReturnCode = " << ret << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    return ret;
}

//
// Set current position as origin (0,0) coordinate of the controller.
//
int XYStage::SetOrigin()
{
    unsigned char sCommand[6] = { 0x6F, MP285::MP285_TxTerm, 0x0A, 0x00, 0x00, 0x00 };
    int ret = WriteCommand(sCommand, 3);

    std::ostringstream osMessage;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::SetOrigin> (ReturnCode=" << ret << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret!=DEVICE_OK) return ret;

    unsigned char sResponse[64];

    memset(sResponse, 0, 64);
    ret = ReadMessage(sResponse, 2);

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::CheckStatus::SetOrigin> (ReturnCode = " << ret << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK) return ret;

    bool yCommError = CheckError(sResponse[0]) != 0;

    char sCommStat[30];
    if (yCommError)
        sprintf(sCommStat, "Error Code ==> <%2x>", sResponse[0]);
    else
        strcpy(sCommStat, "Success");

    //ret = SetProperty(MP285::Instance()->GetMPStr(MP285::MPSTR_CommStateLabel).c_str(), sCommStat);

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}



///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int XYStage::OnSpeed(MM::PropertyBase* /*pProp*/, MM::ActionType /*eAct*/)
{
    return DEVICE_OK;
}

int XYStage::OnGetPositionX(MM::PropertyBase* pProp, MM::ActionType /*eAct*/)
{
    std::ostringstream osMessage;
    int ret = DEVICE_OK;
    double dPosX = MP285::Instance()->GetPositionX();
    double dPosY = MP285::Instance()->GetPositionY();

	osMessage.str("");

    //if (eAct == MM::BeforeGet)
    //{
    //    pProp->Set(dPosX);
	//
	//	if (MP285::Instance()->GetDebugLogFlag() > 1)
	//	{
	//		osMessage << "<MP285Ctrl::OnGetPositionX> BeforeGet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str() << " = [" << dPosX << "], ReturnCode = " << ret;
	//		//this->LogMessage(osMessage.str().c_str());
	//	}
    //}
    //if (eAct == MM::AfterSet)
    //{
        // pProp->Get(dPos);

        ret = GetPositionUm(dPosX, dPosY);
		long lPosX = (long)(dPosX*(double)MP285::Instance()->GetUm2UStep());
		//char sPosX[20];
		//sprintf(sPosX, "%ld", (long)dPosX);

        pProp->Set(lPosX);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnGetPositionX> AfterSet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionX).c_str() << " = [" << dPosX << "," << lPosX << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << ")";
			this->LogMessage(osMessage.str().c_str());
		}

		if (ret != DEVICE_OK) return ret;
    //}

    return DEVICE_OK;
}

int XYStage::OnGetPositionY(MM::PropertyBase* pProp, MM::ActionType /*eAct*/)
{
    std::ostringstream osMessage;
    int ret = DEVICE_OK;
    double dPosX = MP285::Instance()->GetPositionX();
    double dPosY = MP285::Instance()->GetPositionY();

	osMessage.str("");

    //if (eAct == MM::BeforeGet)
    //{        
    //    pProp->Set(dPosY);
	//
	//	if (MP285::Instance()->GetDebugLogFlag() > 1)
	//	{
	//		osMessage << "<MP285Ctrl::OnGetPositionY> BeforeGet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str() << " = [" << dPosY << "], ReturnCode = " << ret;
	//		//this->LogMessage(osMessage.str().c_str());
	//	}
    //}
    //if (eAct == MM::AfterSet)
    //{
        // pProp->Get(dPos)

        ret = GetPositionUm(dPosX, dPosY);
		long lPosY = (long)(dPosY*(double)MP285::Instance()->GetUm2UStep());
		char sPosY[20];
		sprintf(sPosY, "%ld", (long)lPosY);

        pProp->Set(sPosY);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnGetPositionY> AfterSet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_GetPositionY).c_str() << " = [" << dPosY << "," << lPosY << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << ")";
			this->LogMessage(osMessage.str().c_str());
		}

		if (ret != DEVICE_OK) return ret;
    //}

    return DEVICE_OK;
}

int XYStage::OnSetPositionX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream osMessage;
    int ret = DEVICE_OK;
    double dPosX = MP285::Instance()->GetPositionX();
    double dPosY = MP285::Instance()->GetPositionY();

	osMessage.str("");

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dPosX);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnSetPositionX> BeforeGet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionX).c_str() << " = [" << dPosX << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dPosX);

		if (MP285::Instance()->GetMotionMode() == 0)
			ret = SetPositionUm(dPosX, dPosY);
		else
			ret = SetRelativePositionUm(dPosX, 0.);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnSetPositionX> AfterSet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionX).c_str() << " = [" << dPosX << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}
    }

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}

int XYStage::OnSetPositionY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream osMessage;
    int ret = DEVICE_OK;
    double dPosX = MP285::Instance()->GetPositionX();
    double dPosY = MP285::Instance()->GetPositionY();

	osMessage.str("");

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dPosY);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnSetPositionY> BeforeGet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionY).c_str() << " = [" << dPosY << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(dPosY);

		if (MP285::Instance()->GetMotionMode() == 0)
			ret = SetPositionUm(dPosX, dPosY);
		else
			ret = SetRelativePositionUm(0., dPosY);

		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<MP285Ctrl::OnSetPositionY> AfterSet(" << MP285::Instance()->GetMPStr(MP285::MPSTR_SetPositionY).c_str() << " = [" << dPosY << "], ReturnCode = " << ret;
			//this->LogMessage(osMessage.str().c_str());
		}
    }

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage << ")";
		this->LogMessage(osMessage.str().c_str());
	}

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Helper, internal methods
///////////////////////////////////////////////////////////////////////////////

//
// Write a coomand to serial port
//
int XYStage::WriteCommand(unsigned char* sCommand, int nLength)
{
    int ret = DEVICE_OK;
    ostringstream osMessage;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<XYStage::WriteCommand> (Command=";
		char sHex[4] = { '\0', '\0', '\0', '\0' };
		for (int n=0; n < nLength; n++)
		{
			MP285::Instance()->Byte2Hex((const unsigned char)sCommand[n], sHex);
			osMessage << "[" << n << "]=<" << sHex << ">";
		}
		osMessage << ")";
		this->LogMessage(osMessage.str().c_str());
	}

	for (int nBytes = 0; nBytes < nLength && ret == DEVICE_OK; nBytes++)
	{
		ret = WriteToComPort(MP285::Instance()->GetSerialPort().c_str(), (const unsigned char*)&sCommand[nBytes], 1);
		CDeviceUtils::SleepMs(1);
	}

    if (ret != DEVICE_OK) return ret;

    return DEVICE_OK;
}

//
// Read a message from serial port
//
int XYStage::ReadMessage(unsigned char* sResponse, int nBytesRead)
{
    // block/wait for acknowledge, or until we time out;
    unsigned int nLength = 256;
    unsigned char sAnswer[256];
    memset(sAnswer, 0, nLength);
    unsigned long lRead = 0;
    unsigned long lStartTime = GetClockTicksUs();

    ostringstream osMessage;

    char sHex[4] = { '\0', '\0', '\0', '\0' };
    int ret = DEVICE_OK;
    bool yRead = false;
    bool yTimeout = false;
    while (!yRead && !yTimeout && ret == DEVICE_OK )
    {
        unsigned long lByteRead;

        const MM::Device* pDevice = this;
        ret = (GetCoreCallback())->ReadFromSerial(pDevice, MP285::Instance()->GetSerialPort().c_str(), (unsigned char *)&sAnswer[lRead], (unsigned long)nLength-lRead, lByteRead);
       
		if (MP285::Instance()->GetDebugLogFlag() > 2)
		{
			osMessage.str("");
			osMessage << "<MP285Ctrl::ReadMessage> (ReadFromSerial = (" << lByteRead << ")::<";
			for (unsigned long lIndx=0; lIndx < lByteRead; lIndx++)
			{
				// convert to hext format
				MP285::Instance()->Byte2Hex(sAnswer[lRead+lIndx], sHex);
				osMessage << "[" << sHex  << "]";
			}
			osMessage << ">";
			this->LogMessage(osMessage.str().c_str());
		}

        // concade new string
        lRead += lByteRead;

        if (lRead > 2)
        {
            yRead = (sAnswer[0] == 0x30 || sAnswer[0] == 0x31 || sAnswer[0] == 0x32 || sAnswer[0] == 0x34 || sAnswer[0] == 0x38) &&
                    (sAnswer[1] == 0x0D) &&
                    (sAnswer[2] == 0x0D);
        }
        else if (lRead == 2)
        {
            yRead = (sAnswer[0] == 0x0D) && (sAnswer[1] == 0x0D);
        }

        yRead = yRead || (lRead >= (unsigned long)nBytesRead);

        if (yRead) break;
        
        // check for timeout
        yTimeout = ((double)(GetClockTicksUs() - lStartTime) / 10000. ) > (double)m_nAnswerTimeoutMs;
        if (!yTimeout) CDeviceUtils::SleepMs(3);
    }

    // block/wait for acknowledge, or until we time out
    // if (!yRead || yTimeout) return DEVICE_SERIAL_TIMEOUT;
    // MP285::Instance()->ByteCopy(sResponse, sAnswer, nBytesRead);
    // if (checkError(sAnswer[0]) != 0) ret = DEVICE_SERIAL_COMMAND_FAILED;

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage.str("");
		osMessage << "<MP285Ctrl::ReadMessage> (ReadFromSerial = <";
	}

	for (unsigned long lIndx=0; lIndx < (unsigned long)nBytesRead; lIndx++)
	{
		sResponse[lIndx] = sAnswer[lIndx];
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			MP285::Instance()->Byte2Hex(sResponse[lIndx], sHex);
			osMessage << "[" << sHex  << ",";
			MP285::Instance()->Byte2Hex(sAnswer[lIndx], sHex);
			osMessage << sHex  << "]";
		}
	}

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		osMessage << ">";
		this->LogMessage(osMessage.str().c_str());
	}

    return DEVICE_OK;
}

//
// check the error code for the message returned from serial communivation
//
int XYStage::CheckError(unsigned char bErrorCode)
{
    // if the return message is 2 bytes message including CR
    unsigned int nErrorCode = 0;
    ostringstream osMessage;

	osMessage.str("");

    // check 4 error code
    if (bErrorCode == MP285::MP285_SP_OVER_RUN)
    {
        // Serial command buffer over run
        nErrorCode = MPError::MPERR_SerialOverRun;       
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == MP285::MP285_FRAME_ERROR)
    {
        // Receiving serial command time out
        nErrorCode = MPError::MPERR_SerialTimeout;       
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == MP285::MP285_BUFFER_OVER_RUN)
    {
        // Serial command buffer full
        nErrorCode = MPError::MPERR_SerialBufferFull;       
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == MP285::MP285_BAD_COMMAND)
    {
        // Invalid serial command
        nErrorCode = MPError::MPERR_SerialInpInvalid;       
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == MP285::MP285_MOVE_INTERRUPTED)
    {
        // Serial command interrupt motion
        nErrorCode = MPError::MPERR_SerialIntrupMove;       
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == 0x0D)
    {
        // read carriage return
        nErrorCode = MPError::MPERR_OK;
 		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }
    else if (bErrorCode == 0x00)
    {
        // No response from serial port
        nErrorCode = MPError::MPERR_SerialZeroReturn;
		if (MP285::Instance()->GetDebugLogFlag() > 1)
		{
			osMessage << "<XYStage::checkError> ErrorCode=[" << MPError::Instance()->GetErrorText(nErrorCode).c_str() << "])";
		}
    }

	if (MP285::Instance()->GetDebugLogFlag() > 1)
	{
		this->LogMessage(osMessage.str().c_str());
	}

    return (nErrorCode);
}

