///////////////////////////////////////////////////////////////////////////////
// FILE:          XT900.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implementation of X-Cite Led Singleton Class
//
// COPYRIGHT:     S3L GmbH 2021

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
// AUTHOR:        Steffen Leidenbach
//                based on XCiteXT600 code from Lon Chu (lonchu@yahoo.com) created on July 2011
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
#include <time.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "XT900.h"
#include "XT900Ctrl.h"
#include "XT900Dev.h"

using namespace std;

XLed* g_pXLed;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

//
// Initialize the MMDevice name
//
MODULE_API void InitializeModuleData()
{
    g_pXLed = XLed::Instance();       // Initiate the XLed instance

    // initialize the controller device name
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_CtrlBoardName).c_str(), MM::GenericDevice, XLed::Instance()->GetXLedStr(XLed::XL_CtrlBoardDesc).c_str());

    
	// initialize the R,S,T,  U,V,W, X,Y,Z LED device names 
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_RLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_RLedDevDesc).c_str());
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_SLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_SLedDevDesc).c_str());
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_TLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_TLedDevDesc).c_str());
	
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_ULedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_ULedDevDesc).c_str());
    RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_VLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_VLedDevDesc).c_str());
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_WLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_WLedDevDesc).c_str());

	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_XLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_XLedDevDesc).c_str());
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_YLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_YLedDevDesc).c_str());
	RegisterDevice(XLed::Instance()->GetXLedStr(XLed::XL_ZLedDevName).c_str(), MM::ShutterDevice, XLed::Instance()->GetXLedStr(XLed::XL_ZLedDevDesc).c_str());
}

//
// Creating the MMDevice
//
MODULE_API MM::Device* CreateDevice(const char* sDeviceName)
{
    // checking for null pointer
    if (sDeviceName == 0) return 0;

	// create controller device
    if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_CtrlBoardName).c_str()) == 0) 
        return new XLedCtrl();
    
	// create LED devices: R,S,T,  U,V,W, X,Y,Z
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_RLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevR);
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_SLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevS);
    if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_TLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevT);
    

    if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_ULedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevU);
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_VLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevV);
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_WLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevW);

        
    if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_XLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevX);
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_YLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevY);
	if (strcmp(sDeviceName, XLed::Instance()->GetXLedStr(XLed::XL_ZLedDevName).c_str()) == 0)
        return new XLedDev(XLed::XL_LedDevZ);

    // device name is not recognized, return null
    return NULL;
}

//
// delete the device --> invoke device destructor
//
MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

//
// General utility function
//
int ClearPort(MM::Device& device, MM::Core& core, const char* sPort)
{
//	core.PurgeSerial(&device, sPort);

    // Clear contents of serial port 
    const int nBufSize = 255;
    unsigned char sClear[nBufSize];                                                        
    unsigned long lRead = nBufSize;                                               
    int ret;

    // reset the communication port buffer
    while ((int) lRead == nBufSize)                                                     
    { 
        // reading from the serial port
        ret = core.ReadFromSerial(&device, sPort, sClear, nBufSize, lRead);

        std::ostringstream sMessage;
        sMessage << "<XLed::ClearPort> port = (" <<  sPort << ") :: clearBuffer(" << lRead << ")  = (" << sClear << ")";
        core.LogMessage(&device, sMessage.str().c_str(), false);

        // verify the read operation
        if (ret != DEVICE_OK) return ret;                                                           
    } 

    // upon successful restting the port
    return DEVICE_OK;                                                           
} 

bool            XLed::m_yInstanceFlag         = false;        // instance flag
bool            XLed::m_yConnected            = false;        // XLED connected flag
int             XLed::m_nDebugLogFlag         = 0;            // debug log flag
XLed*           XLed::m_pXLed                 = NULL;         // single copy MP285
std::string     XLed::m_sPort;                                // serial port symbols
unsigned char   XLed::m_sParamData[2000];                     // parameter data


XLed::XLed()
{
    XLed::m_sXLedStr[XLed::XL_CtrlBoardName]            = "XT900/NOVEM";	                         //  controller device name
    XLed::m_sXLedStr[XLed::XL_CtrlBoardDesc]            = "Controller";                              //  controller device name

    XLed::m_sXLedStr[XLed::XL_RLedDevName]              = "LED1 Device";                              //  R LED device name
    XLed::m_sXLedStr[XLed::XL_SLedDevName]              = "LED2 Device";                              //  S LED device name
    XLed::m_sXLedStr[XLed::XL_TLedDevName]              = "LED3 Device";                              //  T LED device name
    XLed::m_sXLedStr[XLed::XL_ULedDevName]              = "LED4 Device";                              //  W LED device name
    XLed::m_sXLedStr[XLed::XL_VLedDevName]              = "LED5 Device";                              //  X LED device name
    XLed::m_sXLedStr[XLed::XL_WLedDevName]              = "LED6 Device";                              //  W LED device name
    XLed::m_sXLedStr[XLed::XL_XLedDevName]              = "LED7 Device";                              //  X LED device name
    XLed::m_sXLedStr[XLed::XL_YLedDevName]              = "LED8 Device";                              //  Y LED device name
    XLed::m_sXLedStr[XLed::XL_ZLedDevName]              = "LED9 Device";                              //  Z LED device name
    
	XLed::m_sXLedStr[XLed::XL_RLedDevDesc]              = "DAPI, FURA-2";                             //  R LED device name
    XLed::m_sXLedStr[XLed::XL_SLedDevDesc]              = "CFP";                                      //  S LED device name
    XLed::m_sXLedStr[XLed::XL_TLedDevDesc]              = "FITC";                                     //  T LED device name
    XLed::m_sXLedStr[XLed::XL_ULedDevDesc]              = "YFP, TRITC, mCherry";                      //  W LED device name
    XLed::m_sXLedStr[XLed::XL_VLedDevDesc]              = "YFP, TRITC, mCherry";                      //  X LED device name
    XLed::m_sXLedStr[XLed::XL_WLedDevDesc]              = "YFP, TRITC, mCherry";                      //  W LED device name
    XLed::m_sXLedStr[XLed::XL_XLedDevDesc]              = "YFP, TRITC, mCherry, Cy5";                 //  X LED device name
    XLed::m_sXLedStr[XLed::XL_YLedDevDesc]              = "Cy5, Cy7";                                 //  Y LED device name
    XLed::m_sXLedStr[XLed::XL_ZLedDevDesc]              = "IR800, FURA-2, Cy7";                       //  Z LED device name
    
	XLed::m_sXLedStr[XLed::XL_XLedSoftVer]              = "v1.00.000";                                //  XLed Software Version
    XLed::m_sXLedStr[XLed::XL_LogFilename]              = /*NULL;*/"XLedLog.txt";                     //  XLed Logfile name
    XLed::m_sXLedStr[XLed::XL_CtrlBoardNameLabel]       = "X.00 Controller";                          //  XLed controller device name label
    XLed::m_sXLedStr[XLed::XL_CtrlBoardDescLabel]       = "X.01 Controller";                          //  XLed controller device decription label
    XLed::m_sXLedStr[XLed::XL_SerialNumberLabel]        = "X.02 Controller Serial Number";            //  XLed controller serial number label
    XLed::m_sXLedStr[XLed::XL_UnitSoftVerLabel]         = "X.03 Software Version";                    //  XLed Software Version Label
    XLed::m_sXLedStr[XLed::XL_XLedSoftVerLabel]         = "X.04 Turbo/XT900 Software Version";        //  XLed Software Version Label
    XLed::m_sXLedStr[XLed::XL_CtrlBoardConnLabel]       = "X.05 Board Connected";                     //  XLed board connected label
    XLed::m_sXLedStr[XLed::XL_XLedDebugLogFlag]         = "X.06 Debug Log Flag";                      //  XLed Software Version
    XLed::m_sXLedStr[XLed::XL_XLedStatusLabel]          = "X.07 Status (0-65535)";                    //  XLed status label
    XLed::m_sXLedStr[XLed::XL_XLedStatusDescLabel]      = "X.07 Status Description";                  //  XLed status description label
    XLed::m_sXLedStr[XLed::XL_AllOnOffLabel]            = "X.08 All On/Off (1=On 0=Off)";             //  All leds ON/OFF label
    XLed::m_sXLedStr[XLed::XL_PWMStateLabel]            = "X.09 IPG State (1=Run 0=Off)";             //  XLed PWM State label
    XLed::m_sXLedStr[XLed::XL_PWMModeLabel]             = "X.10 IPG Running Mode (1=Sing 0=Cont)";    //  XLed PWM mode label
    XLed::m_sXLedStr[XLed::XL_FrontPanelLabel]          = "X.11 Front Panel (1=Lock 0=Unlock)";       //  Front Panel (Lock/UnLock) label
    XLed::m_sXLedStr[XLed::XL_LCDScrnNumberLabel]       = "X.12 LCD Screen Number (2-6;8-13)";        //  LCD screen number label
    XLed::m_sXLedStr[XLed::XL_LCDScrnBriteLabel]        = "X.13 LCD Brightness (0-255)";              //  LCD Screen brightness label
    XLed::m_sXLedStr[XLed::XL_LCDScrnSaverLabel]        = "X.14 LCD Screen Timeout (0:Disabled)";     //  LCD screen saver timeout label
    XLed::m_sXLedStr[XLed::XL_ClearAlarmLebel]          = "X.15 Clear Alarm (1:Cleared 0:Ok)";        //  Clear Alarm label
    XLed::m_sXLedStr[XLed::XL_SpeakerVolumeLabel]       = "X.16 Speaker Volume (0-255)";              //  Speaker volume label
    XLed::m_sXLedStr[XLed::XL_LedDevNameLabel]          = "L.00 Device ";                             //  Led device name label
    XLed::m_sXLedStr[XLed::XL_LedDevDescLabel]          = "L.01 Device ";                             //  Led device description
    //XLed::m_sXLedStr[XLed::XL_LedDevTypeLabel]          = "L.02 Device Type";                       //  Led device type label
    //XLed::m_sXLedStr[XLed::XL_LedSerialNoLabel]         = "L.02 Serial Number";                     //  Led device serial number label
    XLed::m_sXLedStr[XLed::XL_LedMfgDateLabel]          = "L.02 Manufacturing Date";                  //  Led device manufacturing date label
    XLed::m_sXLedStr[XLed::XL_LedWaveLengthLabel]       = "L.03 WaveLength";                          //  Led device wavelength label
    XLed::m_sXLedStr[XLed::XL_LedFWHMLabel]             = "L.04 FWHM Value";                          //  Led device FWHM value label
    XLed::m_sXLedStr[XLed::XL_LedStatusLabel]           = "L.05 Status Byte (0-255)";                 //  Led device status Label
    XLed::m_sXLedStr[XLed::XL_LedStatusDescLabel]       = "L.06 Status Description";                  //  Led device status description Label
    XLed::m_sXLedStr[XLed::XL_LedHoursLabel]            = "L.07 Hours Elapsed";                       //  Led device number of hours label
	XLed::m_sXLedStr[XLed::XL_LedMinIntensityLabel]		= "L.08 Minimum Intensity";					  //  Led Minimum Intensity
    //XLed::m_sXLedStr[XLed::XL_LedPulseWidthLabel]       = "L.11 Minimum Pulse Width";                 //  Led device min pulse width label
    //XLed::m_sXLedStr[XLed::XL_PulseModeLabel]           = "L.12 Pulse Mode (0:DC 1:Int 2:Ext 3:Glbl)";//  XLed pulse mode lael
    //XLed::m_sXLedStr[XLed::XL_SignalOnTimeLabel]        = "L.13 Signal On Time (0-65535)";            //  Led device signal on time label
    //XLed::m_sXLedStr[XLed::XL_SignalOffTimeLabel]       = "L.14 Signal Off Time (0-65535)";           //  Led device signal off time label
    //XLed::m_sXLedStr[XLed::XL_SignalDelayTimeLabel]     = "L.15 Signal Delay Time (0-65535)";         //  Led device signal delay time label
    //XLed::m_sXLedStr[XLed::XL_TriggerDelayTimeLabel]    = "L.16 Trigger Delay Time (0-65535)";        //  Led device trigger delay time label
    //XLed::m_sXLedStr[XLed::XL_PWMUnitsLabel]            = "L.17 IPG Units (0:uS/1:mS/2:S)";           //  Led device PWM units label
    XLed::m_sXLedStr[XLed::XL_LedTempLabel]             = "L.09 Current Temperature (Deg.C)";         //  Led device temperature label
    XLed::m_sXLedStr[XLed::XL_LedMaxTempLabel]          = "L.10 Max Allowed Temperature (Deg.C)";     //  Led device max allowed temperature label [°C]
    XLed::m_sXLedStr[XLed::XL_LedMinTempLabel]          = "L.11 Min Allowed Temperature (Deg.C)";     //  Led device min allowed temperature label [°C]
    XLed::m_sXLedStr[XLed::XL_LedTempHystLabel]         = "L.12 Temperature Hysteresis (Deg.C)";      //  Led device temperature hysteresis label [°C]
	XLed::m_sXLedStr[XLed::XL_LedTriggerSequenceLabel]	= "L.13 Trigger Sequence";					  //  TTL Trigger sequence
    XLed::m_sXLedStr[XLed::XL_LedOnOffStateLabel]       = "L.14 On/Off State (1=On 0=Off)";           //  Led device ON/OFF state label
    XLed::m_sXLedStr[XLed::XL_LedIntensityLabel]        = "L.15 Intensity (0.0 or ";//5.0 - 100.0)%"; //  Led device intensity label
    XLed::m_sXLedStr[XLed::XL_Reserved]                 = "Reserved";                                 //  Led device software version label
}

XLed::~XLed()
{
    if (m_pXLed) delete m_pXLed;
	m_pXLed = NULL;
    m_yInstanceFlag = false;
}

XLed* XLed::Instance()
{
    if(!m_yInstanceFlag)
    {
        m_pXLed = new XLed();
        m_yInstanceFlag = true;
        memset(m_sParamData, 0, 2000);
    }

    return m_pXLed;
}

//
// Get XLed constant string
//
std::string XLed::GetXLedStr(int nXLedStrCode) const
{ 
   string sText;        // XLed String

   if (m_pXLed != NULL)
   {
       map<int, string>::const_iterator nIterator;
       nIterator = m_sXLedStr.find(nXLedStrCode);   
       if (nIterator != m_sXLedStr.end())
          sText = nIterator->second;
   }

   return sText;
}

//
// Copy byte data buffer for iLength
//
int XLed::ByteCopy(unsigned char* bDst, const unsigned char* bSrc, int nLength)
{
    int nBytes = 0;
    if (bSrc == NULL || bDst == NULL) return(nBytes);
    for (nBytes = 0; nBytes < nLength; nBytes++) bDst[nBytes] = bSrc[nBytes];
    return nBytes;
}

//
// Convert byte data to hex string
//
void XLed::Byte2Hex(const unsigned char bByte, char* sHex)
{
    char sHexDigit[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
    sHex[2] =  '\0';
    sHex[1] = sHexDigit[(int)(bByte & 0xF)];
    sHex[0] = sHexDigit[(int)(bByte / 0x10)];
    return;
}

//
// Get address of the parameter ID
//
unsigned char* XLed::GetParameter(int nParameterID)
{
    if (nParameterID < 0) return NULL;
    if (nParameterID >= XL_MaxParameters) return NULL;
    return &m_sParamData[nParameterID * XL_MaxPropSize];
}
