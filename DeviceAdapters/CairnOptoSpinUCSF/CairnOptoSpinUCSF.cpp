///////////////////////////////////////////////////////////////////////////////
// FILE:       CairOptoSpinUCSF.cpp
// PROJECT:    MicroManager
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:
// Conix adapter
//                
// AUTHOR: Nico Stuurman, 09/23/2025
//

#ifdef WIN32
#include <windows.h>
#endif

#include "CairnOptospinUCSF.h"
#include <cstdio>
#include <string>
#include "ModuleInterface.h"
#include <sstream>
#include <iostream>
#include "ftd2xx.h"

const char* g_CairnOptospinName = "CairnOptospin";

using namespace std;



///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	// We look for VID 156B, PID 0003
	// Description: "Cairn Optospin"

	FT_STATUS ftStatus;
	DWORD numDevs = 0;
	FT_HANDLE ftHandleTemp;
	DWORD Flags;
	DWORD ID;
	DWORD Type;
	DWORD LocId;
	char SerialNumber[16];
	char Description[64];

	// create the device information list
	ftStatus = FT_CreateDeviceInfoList(&numDevs);
	if (ftStatus == FT_OK) {
		printf("Number of devices is %d\n", numDevs);
	}
	printf("Now enumerating....\n");
	if (numDevs > 0) {
		for (DWORD i = 0; i < numDevs; i++)
		{
			ftStatus = FT_GetDeviceInfoDetail(i, &Flags, &Type, &ID, &LocId, SerialNumber,
				Description, &ftHandleTemp);
			if (ftStatus == FT_OK) {
            /*
				printf("Dev:%x\n", i);
				printf(" Flags=0x%x\n", Flags);
				printf(" Type=0x%x\n", Type);
				printf(" ID=0x%x\n", ID);
				printf(" LocId=0x%x\n", LocId);
				printf(" SerialNumber=%s\n", SerialNumber);
				printf(" Description=%s\n", Description);
				printf(" ftHandle=0x%x\n", ftHandleTemp);
            */

				if (ID == 0x156b0003)
				{
					printf("Found Cairn OptoSpin\n");
               std::ostringstream os;
               os << g_CairnOptospinName << "_" << i;
               RegisterDevice(os.str().c_str(), MM::StateDevice, "CairnOptospin");
				}
			}
		}
	}

}



MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0) {
		return 0;
	}
   std::string dev = deviceName;
   std::string prefix = g_CairnOptospinName;
 
   if (dev.length() < prefix.length()) {
      return 0;
   }
   if (dev.compare(0, prefix.length(), prefix) == 0)
   {
      size_t pos = dev.find_last_of('_');
      if (pos != std::string::npos) {
         std::string strId = dev.substr(pos + 1);
         try {
            DWORD devID = std::stoi(strId);
            CairnOptospin* pC = new CairnOptospin(devID);
            return pC;
         } catch (const std::invalid_argument&) {
            printf("Invalid ID");
         }
         catch (const std::out_of_range&) {
            printf("Out of range");
         }
      }
   }

	return 0;
}



MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}





///////////////////////////////////////////////////////////////////////////////
// CairnOptospin device
///////////////////////////////////////////////////////////////////////////////

CairnOptospin::CairnOptospin(DWORD devId) :
   initialized_(false),
   numPos_(6),
   ftHandle_(0),
   devId_(devId)
{
   InitializeDefaultErrorMessages();

   // create pre-initialization properties
   // ------------------------------------

   // Name
   std::ostringstream os;
   os << g_CairnOptospinName << "_" << devId_;
   CreateProperty(MM::g_Keyword_Name, os.str().c_str(), MM::String, true);

   // Description
   CreateProperty(MM::g_Keyword_Description, "Cairn Optospin - UCSF Device adapter", MM::String, true);

   LogMessage("Loaded Cairn UCSG Device adapter", false);
}



CairnOptospin::~CairnOptospin()
{
   Shutdown();
}



void CairnOptospin::GetName(char* Name) const
{
   std::ostringstream os;
   os << g_CairnOptospinName << "_" << devId_;
   CDeviceUtils::CopyLimitedString(Name, os.str().c_str());
}


int CairnOptospin::Initialize()
{
   FT_STATUS ftStatus = FT_Open(devId_, &ftHandle_);
   if (ftStatus != FT_OK) {
      return DEVICE_NOT_CONNECTED;
   }

   initialized_ = true;

   // set property list
   // -----------------
   
   // Get and Set State (allowed values 1-4, start at 0 for Hardware Configuration Wizard))
   CPropertyAction *pAct = new CPropertyAction (this, &CairnOptospin::OnState);
   int nRet=CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
   if (nRet != DEVICE_OK)
      return nRet;
   for (long i = 0; i < numPos_; i++) {
      AddAllowedValue(MM::g_Keyword_State, std::to_string(i).c_str());
   }


   // Label
   // -----
   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   nRet = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
   if (nRet != DEVICE_OK)                                                     
      return nRet;   

   // create default positions and labels
   for (long i=0; i < numPos_; i++)
   {
      std::ostringstream os;
      os << "Position_" << i;
      SetPositionLabel(i, os.str().c_str());
   }

   // nRet = UpdateStatus();
   if (nRet != DEVICE_OK)
      return nRet;

   return DEVICE_OK;
}



int CairnOptospin::Shutdown()
{
   if (initialized_)
   {
      if (ftHandle_ != 0) {
         FT_Close(ftHandle_);
      }
      initialized_ = false;
   }
   return DEVICE_OK;
}

/*
 *
 */
bool CairnOptospin::Busy()
{
   // the commands are blocking, so we cannot be busy
   return false;
   /*
   const unsigned long answerLength = 40;
   // If there was no command pending we are not busy
   if (!pendingCommand_)
      return false;

   // Read a line from the port, if first char is 'A' we are OK
   // if first char is 'N' read the error code
   unsigned char answer[answerLength];
   unsigned long charsRead;
   ReadFromComPort(port_.c_str(), answer, answerLength, charsRead);
   if (answer[0] == "A") {
      // this command was finished and is not pending anymore
      pendingCommand_ = false;
      return true;
   }
   else
      return false;
   
   // we should never be here, better not to block
   return false;
   */
}



///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////


int CairnOptospin::GetDevicePosition(int& position) 
{

   position = 1;
   return DEVICE_OK;
}



int CairnOptospin::SetDevicePosition(int position)
{
   return DEVICE_OK;
}



// Needs to be worked on (a lot)
int CairnOptospin::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      int position, ret;
      ret = GetDevicePosition(position);
      if (ret != DEVICE_OK)
         return ret;
      pProp->Set((long) position - 1);
   }
   else if (eAct == MM::AfterSet) {
      long position;
      int ret;
      pProp->Get(position);
      ret = SetDevicePosition((int) position + 1);
      if (ret != DEVICE_OK)
         return ret;
   }
   return DEVICE_OK;
}

