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

const char* g_CairnOptospinName = "OptospinWheel";
const char* g_CairnOptospinHubName = "CairnOptospinController";

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
      for (DWORD i = 0; i < numDevs; i++)
      {
         ftStatus = FT_GetDeviceInfoDetail(i, &Flags, &Type, &ID, &LocId, SerialNumber,
            Description, &ftHandleTemp);
         if (ftStatus == FT_OK)
         {
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
               RegisterDevice(g_CairnOptospinHubName, MM::HubDevice, "CairnOptospinController");
               return;
            }
         }
      }
   }
}



MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (strcmp(deviceName, g_CairnOptospinHubName) == 0)
   {
      return new CairnHub();
   } else if (strncmp(deviceName, g_CairnOptospinName, strlen(g_CairnOptospinName)) == 0)
   {
      // deviceName is expected to be of the formn "OptospinWheel_#" where # is the wheel number
      int wheelNumber = 0;
      const char* underscore = strchr(deviceName, '_');
      if (underscore != 0)
      {
         wheelNumber = atoi(underscore + 1);
      }
      if (wheelNumber >= 1 && wheelNumber <= 4)
      {
         return new CairnOptospin(wheelNumber);
      }
   }

	return 0;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


CairnHub::CairnHub() :
   ftHandle_(0),
   initialized_(false)
{
   for (int i = 0; i < 4; i++) {
      filterWheels_[i] = false;
   }

   // Create serial number pre-initialization property
   // We need to query for available Controllers based on VID and PID and 
   // then populate the SerialNumber property
   CPropertyAction* pAct = new CPropertyAction(this, &CairnHub::OnSerialNumber);
   CreateProperty("SerialNumber", "0", MM::String, false, pAct, true);

   // We look for VID 156B, PID 0003
   // Description: "Cairn Optospin"
   FT_STATUS ftStatus;
   DWORD numDevs = 0;
   FT_HANDLE ftHandleTemp;
   DWORD Flags;
   DWORD ID;
   DWORD DType;
   DWORD LocId;
   char SerialNumber[16];
   char Description[64];

   // create the device information list
   ftStatus = FT_CreateDeviceInfoList(&numDevs);
   if (ftStatus == FT_OK) {
      for (DWORD i = 0; i < numDevs; i++)
      {
         ftStatus = FT_GetDeviceInfoDetail(i, &Flags, &DType, &ID, &LocId, SerialNumber,
            Description, &ftHandleTemp);
         if (ftStatus == FT_OK)
         {
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
               AddAllowedValue("SerialNumber", SerialNumber);
            }
         }
      }
   }
}

CairnHub::~CairnHub()
{
   if (initialized_)
   {
      Shutdown();
   }

}

int CairnHub::Shutdown()
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


int CairnHub::Initialize()
{
   // We look for VID 156B, PID 0003
   // Description: "Cairn Optospin"
   int devId = -1;
   FT_STATUS ftStatus;
   DWORD numDevs = 0;
   FT_HANDLE ftHandleTemp;
   DWORD Flags;
   DWORD ID;
   DWORD DType;
   DWORD LocId;
   char SerialNumber[16];
   char Description[64];

   // create the device information list
   ftStatus = FT_CreateDeviceInfoList(&numDevs);
   if (ftStatus == FT_OK) {
      for (DWORD i = 0; i < numDevs; i++)
      {
         ftStatus = FT_GetDeviceInfoDetail(i, &Flags, &DType, &ID, &LocId, SerialNumber,
            Description, &ftHandleTemp);
         if (ftStatus == FT_OK)
         {
            if (ID == 0x156b0003)
            {
               if (strcmp(SerialNumber, serialNumber_.c_str()) == 0)
               {
                  devId = i;
               }
            }
         }
      }
   }
   if (devId == -1)
   {
      return DEVICE_NOT_CONNECTED;
   }

   // Open FT device handle
   ftStatus = FT_Open(devId, &ftHandle_);
   if (ftStatus != FT_OK) {
      return DEVICE_NOT_CONNECTED;
   }

   initialized_ = true;

   // Sets read and write timeouts in milliseconds
   ftStatus = FT_SetTimeouts(ftHandle_, 25, 25);

   // Get the software version of the CairnOptoSpin
   DWORD bytesWritten;
   unsigned char txBuffer[2]; // Contains data to write to device
   txBuffer[0] = 0;
   txBuffer[1] = 0x40;
   ftStatus = FT_Write(ftHandle_, txBuffer, sizeof(txBuffer), &bytesWritten);
   if (ftStatus == FT_OK) 
   {
      if (bytesWritten == sizeof(txBuffer)) 
      {
         unsigned char answer[4];
         DWORD bytesRead;
         ftStatus = FT_Read(ftHandle_, answer, sizeof(answer), &bytesRead);
         if (ftStatus == FT_OK)
         {
            if (answer[0] == 0xFF)
            {
               if (answer[1] == 2) {
                  // answer is big endian, convert to little, will be wrong on big endian host
                  uint16_t softwareVersion;
                  memcpy(&softwareVersion, &answer[2], sizeof(uint16_t));
                  softwareVersion_ = (softwareVersion >> 8) | (softwareVersion << 8);
                  std::ostringstream oss;
                  oss << "Software version of Optospin is: " << softwareVersion_;
                  this->LogMessage(oss.str().c_str(), false);
               }
            }
         }
      }
   }

   // Get the number of filter wheels connected to the controller
   txBuffer[0] = 0;
   txBuffer[1] = 0x54;
   ftStatus = FT_Write(ftHandle_, txBuffer, sizeof(txBuffer), &bytesWritten);
   if (ftStatus == FT_OK)
   {
      if (bytesWritten == sizeof(txBuffer))
      {
         unsigned char answer[3];
         DWORD bytesRead;
         ftStatus = FT_Read(ftHandle_, answer, sizeof(answer), &bytesRead);
         if (ftStatus == FT_OK)
         {
            if (answer[0] == 0xFF)
            {
               if (answer[1] == 1) {
                  // This returns a single byte, in which the least significant bit (bit 0) is set if filterwheel 1 is present, 
                  // bit 1 is set if filterwheel 2 is present, bit 2 is set if filterwheel 3 is present and
                  // bit 3 is set if filterwheel 4 is present.
                  filterWheels_[0] = answer[2] & 0x1;
                  filterWheels_[1] = (answer[2] >> 1) & 0x1;
                  filterWheels_[2] = (answer[2] >> 2) & 0x1;
                  filterWheels_[3] = (answer[2] >> 3) & 0x1;
               }
            }
         }
      }
   }
   if (ftStatus != FT_OK) 
   {
      return ERR_UNKNOWN_COMMAND;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &CairnHub::OnSoftwareVersion);
   CreateProperty("SoftwareVersion", "0", MM::Integer, true, pAct);

   pAct = new CPropertyAction(this, &CairnHub::OnSerialNumber);
   CreateProperty("Controller_SerialNumber", "0", MM::String, false, pAct);

   return DEVICE_OK;
}

void CairnHub::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_CairnOptospinHubName);
}

bool CairnHub::Busy()
{
   long position;
   return GetWheelPosition(1, position) == ERR_OPTOSPIN_BUSY;
}

int CairnHub::DetectInstalledDevices()
{
   bool initialized = initialized_;
   if (!initialized) {
      int ret = Initialize();
      if (ret != DEVICE_OK) {
         return ret;
      }
   }
   for (int i = 0; i < 4; i++) {
      if (filterWheels_[i]) {
         AddInstalledDevice(new CairnOptospin(i + 1));
      }
   }
   if (!initialized) {
      Shutdown();
   }
   return DEVICE_OK;
}

int CairnHub::GetWheelPosition(long wheelNumber, long& position)
{
   if (wheelNumber < 1 || wheelNumber > 4) {
      return ERR_INVALID_WHEEL_NUMBER;
   }
   if (!filterWheels_[wheelNumber - 1]) {
      return ERR_WHEEL_NOT_CONNECTED;
   }

   DWORD bytesWritten;
   uint8_t txBuffer[2]; // Contains data to write to device
   txBuffer[0] = 0;
   txBuffer[1] = 0x9c;
   FT_STATUS ftStatus = FT_Write(ftHandle_, txBuffer, sizeof(txBuffer), &bytesWritten);
   if (ftStatus == FT_OK) 
   {
      if (bytesWritten == sizeof(txBuffer)) 
      {
         unsigned char answer[2];
         DWORD bytesRead;
         ftStatus = FT_Read(ftHandle_, answer, sizeof(answer), &bytesRead);
         if (ftStatus == FT_OK && bytesRead == 2)
         {
            if (answer[0] == 0xFF)
            {
               if (answer[1] == 4)
               {
                  // controller responds with position of all wheels in 4 bytes
                  unsigned char answer2[4];
                  ftStatus = FT_Read(ftHandle_, answer2, sizeof(answer2), &bytesRead);
                  if (ftStatus == FT_OK && bytesRead == 4)
                  {
                     position = answer2[wheelNumber - 1];
                  }
               }
               return DEVICE_OK;
            }
            else if (answer[0] == 4 || answer[0] == 7)
            {
               return ERR_OPTOSPIN_BUSY;
            }
         }
      }
   }
   if (ftStatus != FT_OK) 
   {
      return ERR_UNKNOWN_COMMAND;
   }
   return DEVICE_OK;
}

/*
 * Sets the filter position of th requested wheel
 * Both wheelNUmber and position are 1-based, so that 
 * no conversion needs to take place in calling the controller
*/
int CairnHub::SetWheelPosition(long wheelNumber, long position)
{
   if (wheelNumber < 1 || wheelNumber > 4) {
      return ERR_INVALID_WHEEL_NUMBER;
   }
   if (!filterWheels_[wheelNumber - 1]) {
      return ERR_WHEEL_NOT_CONNECTED;
   }
   if (position < 1 || position > 6) {
      return ERR_INVALID_POSITION;
   }
   DWORD bytesWritten;
   uint8_t txBuffer[6]; // Contains data to write to device
   txBuffer[0] = 0;
   txBuffer[1] = 0x8c;
   for (int i = 2; i < 6; i++) {
      txBuffer[i] = 0;
   }
   txBuffer[1 + wheelNumber] = (uint8_t) position;
   FT_STATUS ftStatus = FT_Write(ftHandle_, txBuffer, sizeof(txBuffer), &bytesWritten);
   if (ftStatus == FT_OK) 
   {
      if (bytesWritten == sizeof(txBuffer)) 
      {
         unsigned char answer[2];
         DWORD bytesRead;
         ftStatus = FT_Read(ftHandle_, answer, sizeof(answer), &bytesRead);
         if (ftStatus == FT_OK)
         {
            if (answer[0] == 0xFF)
            {
               return DEVICE_OK;
            }
            else if (answer[0] == 4 || answer[0] == 7)
            {
               return ERR_OPTOSPIN_BUSY;
            }
            return DEVICE_ERR;
         }
      }
   }
   if (ftStatus != FT_OK) 
   {
      return ERR_UNKNOWN_COMMAND;
   }
   return DEVICE_OK;
}

// action interface
// ----------------
int CairnHub::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(serialNumber_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(serialNumber_);

   }
   return DEVICE_OK;
}

int CairnHub::OnSoftwareVersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(softwareVersion_);
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(softwareVersion_);
   }
   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// CairnOptospin device
///////////////////////////////////////////////////////////////////////////////

CairnOptospin::CairnOptospin(long wheelNumber) :
   initialized_(false),
   numPos_(6),
   serialNumber_(0),
   wheelNumber_(wheelNumber)
{
   InitializeDefaultErrorMessages();

   // create pre-initialization properties
   // ------------------------------------

   // Name
   std::ostringstream os;
   os << g_CairnOptospinName << "_" << wheelNumber_;
   CreateProperty(MM::g_Keyword_Name, os.str().c_str(), MM::String, true);

   // Description
   CreateProperty(MM::g_Keyword_Description, "Cairn Optospin - UCSF Device adapter", MM::String, true);

   LogMessage("Loaded Cairn OptoSpin UCSF Device adapter", false);
}

CairnOptospin::~CairnOptospin()
{
   Shutdown();
}

void CairnOptospin::GetName(char* pszName) const
{
   std::ostringstream os;
   os << g_CairnOptospinName << "_" << wheelNumber_;
   CDeviceUtils::CopyLimitedString(pszName, os.str().c_str());
}

int CairnOptospin::Initialize()
{
   // State
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

   initialized_ = true;

   return DEVICE_OK;
}



int CairnOptospin::Shutdown()
{
   // nothing to clean up
   return DEVICE_OK;
}

bool CairnOptospin::Busy()
{
   return this->GetParentHub()->Busy();
}



///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////


int CairnOptospin::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   CairnHub* hub = (CairnHub*) this->GetParentHub();
   if (hub == 0)
      return DEVICE_ERR;
   long position;
   if (eAct == MM::BeforeGet) {
      int ret = hub->GetWheelPosition(wheelNumber_, position);
      if (ret != DEVICE_OK)
         return ret;
      pProp->Set(position - 1);
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(position);
      return  hub->SetWheelPosition(wheelNumber_, position + 1);
   }
   return DEVICE_OK;
}


int CairnOptospin::OnWheelNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(wheelNumber_);
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(wheelNumber_);
   }
   return DEVICE_OK;
}
