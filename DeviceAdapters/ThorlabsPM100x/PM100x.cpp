///////////////////////////////////////////////////////////////////////////////
// FILE:          PM100x.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters - Thorlabs PM100x adapter
//-----------------------------------------------------------------------------
// DESCRIPTION:   This device adapter interfaces with Thorlabs light power meters
//
//                
// AUTHOR:        Nico Stuurman, Altos Labs, 2022
//
// COPYRIGHT:     Altos Labs Inc., 2022
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.


#include "PM100x.h"

#ifndef VI_ERROR_RSRC_NFOUND
#define VI_ERROR_RSRC_NFOUND 111
#endif


const char* g_PM100Name = "ThorlabsPM100";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_PM100Name, MM::GenericDevice, "ThorLabs PowerMeters");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_PM100Name) == 0)
   {
      return new PM100();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}



////////////////////////////////////////////////////////////////////////////////
// PM100 Device
///////////////////////////////////////////////////////////////////////////////
PM100::PM100() :
   initialized_(false),
   deviceName_(""),
   instrHdl_(VI_NULL)
{ 
   std::vector<std::string> pmNames;
   pmNames.push_back("");
   int result = FindPowerMeters(pmNames);

   pmNames.size() > 1 ? deviceName_ = pmNames.at(1) : deviceName_ = pmNames.at(0);
   
   // custom error messages
   SetErrorText(ERR_NO_PM_CONNECTED, "No Power Meters connected");

   // create pre-initialization properties
   // ------------------------------------

   // Port
   CPropertyAction* pAct = new CPropertyAction(this, &PM100::OnPMName);
   CreateProperty("PowerMeter", deviceName_.c_str(), MM::String, false, pAct, true);
   for (size_t cnt = 0; cnt < pmNames.size(); cnt++)
   {
      AddAllowedValue("PowerMeter", pmNames.at(cnt).c_str());
   }

}

PM100::~PM100() 
{
}


int PM100::Initialize()
{

   std::vector<std::string> pmNames;
   int result = FindPowerMeters(pmNames);
   if (result != DEVICE_OK)
      return result;

   if (pmNames.size() < 1)
      return ERR_NO_PM_CONNECTED;

   if (std::find(pmNames.begin(), pmNames.end(), deviceName_) == pmNames.end())
      return ERR_PM_NOT_CONNECTED;

   // deviceName_ = pmNames.at(0);

   // the TLPM lib seems to hold on to the rsrcName, so it 
   static ViChar rsrcName[TLPM_BUFFER_SIZE];

   CDeviceUtils::CopyLimitedString(rsrcName, deviceName_.c_str());

   ViStatus err = TLPM_init(rsrcName, VI_ON, VI_OFF, &instrHdl_);
   if (err != VI_SUCCESS) {
      return (int) err;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &PM100::OnValue);
   int ret = CreateProperty("Power", "0", MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;
   return DEVICE_OK;
}


int PM100::Shutdown() 
{
   if (initialized_ && instrHdl_ != VI_NULL)
   {
      ViStatus err = TLPM_close(instrHdl_);
      if (err != VI_SUCCESS)
         return (int)err;
   }
   return DEVICE_OK;
}


void PM100::GetName(char* pszName) const {
   CDeviceUtils::CopyLimitedString(pszName, deviceName_.c_str());
}


bool PM100::Busy() {return false;}


// action interface
// ----------------
int PM100::OnValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      ViStatus       err = VI_SUCCESS;
      ViReal64       power = 0.0;
      ViInt16        power_unit;
      std::string unit;

      err = TLPM_getPowerUnit(instrHdl_, &power_unit);
      switch (power_unit)
      {
         case TLPM_POWER_UNIT_DBM: unit = "dBm"; break;
         default: unit = "W"; break;
      }
      if (!err) 
         err = TLPM_measPower(instrHdl_, &power);
      if (!err)
      {
         if (unit == "W")
         {
            if (power < 1e-9)
            {
               unit = "nW";
               power *= 1e9;
            }
            else if (power < 1e-6)
            {
               unit = "uW";
               power *= 1e6;
            }
            else if (power < 1e-3)
            {
               unit = "mW";
               power *= 1e3;
            }
         }
         std::ostringstream os;
         os << power << unit;
         pProp->Set(os.str().c_str());
         return DEVICE_OK;
      }
   }
   return DEVICE_ERR;
}


int PM100::OnPMName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceName_.c_str());
   }
   if (eAct == MM::AfterSet)
   {
      pProp->Get(deviceName_);
   }
   return DEVICE_OK;
}


int PM100::FindPowerMeters(std::vector<std::string>& deviceNames)
{
   ViUInt32 deviceCount;
   ViChar rsrcDescr[TLPM_BUFFER_SIZE];
   rsrcDescr[0] = '\0';

   int err = TLPM_findRsrc(0, &deviceCount);
   switch (err)
   {
      case VI_SUCCESS:
         // At least one device was found. Nothing to do here. Continue with device selection menu.
         break;

      case VI_ERROR_RSRC_NFOUND:
         printf("No matching instruments found\n\n");
         return (err);

      default:
         return (err);
   }

   for (ViUInt32 cnt = 0; cnt < deviceCount; cnt++)
   {
      err = TLPM_getRsrcName(0, 0, rsrcDescr);
      if (err != VI_SUCCESS)
         return err;
      deviceNames.push_back(rsrcDescr);
   }

   return DEVICE_OK;
}