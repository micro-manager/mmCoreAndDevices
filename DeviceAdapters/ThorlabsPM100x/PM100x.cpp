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
   instrHdl_(VI_NULL),
   deviceName_("")
{
   rscPtr_ = new ViChar();
}

PM100::~PM100() 
{
   delete rscPtr_;
}


int PM100::Initialize()
{
   std::vector<std::string> deviceNames;

   ViStatus err = FindInstruments(PM100USB_FIND_PATTERN, deviceNames);
   if (err != VI_SUCCESS) {
      return (int) err;
   }

   if (deviceNames.size() > 0)
      deviceName_ = deviceNames.at(0);

   CDeviceUtils::CopyLimitedString(rscPtr_, deviceName_.c_str());

   err = TLPM_init(rscPtr_, VI_ON, VI_OFF, &instrHdl_);
   if (err != VI_SUCCESS) {
      return (int)err;
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
      ViInt16        power_unit = TLPM_POWER_UNIT_WATT;
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


int PM100::OnDeviceName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceName_.c_str());
   }
   return DEVICE_OK;
}


ViStatus PM100::FindInstruments(ViString findPattern, std::vector<std::string>& deviceNames)
{
 
   ViStatus err;
   ViUInt32 deviceCount;
   ViChar rsrcDescr[TLPM_BUFFER_SIZE];
   ViChar name[TLPM_BUFFER_SIZE], sernr[TLPM_BUFFER_SIZE];
   ViBoolean available;

   // prepare return value
   rsrcDescr[0] = '\0';

   err = TLPM_findRsrc(0, &deviceCount);
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

   for (int i = 0; i < deviceCount; i++)
   {
      err = TLPM_getRsrcName(0, i, rsrcDescr);
      deviceNames.push_back(rsrcDescr);
      if (err != VI_SUCCESS)
         return err;
   }

   return VI_SUCCESS;
   /*

   if (deviceCount < 2)
   {
      // Found only one matching instrument - return this
      err = TLPM_getRsrcName(0, 0, rsrcDescr);
      deviceName_ = rsrcDescr;
      return (err);
   }

   // Found multiple instruments - Display list of instruments
   done = 0;
   do
   {
      // Print device list
      for (cnt = 0; cnt < deviceCount; cnt++)
      {
         err = TLPM_getRsrcInfo(0, cnt, name, sernr, VI_NULL, &available);
         if (err) return (err);
         printf("% d(%s): S/N:%s \t%s\n", cnt + 1, (available) ? "FREE" : "LOCK", sernr, name);
      }

      printf("\nPlease select, press q to exit: ");
      while ((i = getchar()) == EOF);
      fflush(stdin);
      switch (i)
      {
      case 'q':
      case 'Q':
         printf("User abort\n\n");
         return (VI_ERROR_RSRC_NFOUND);

      default:
         break;   // do nothing
      }

      // an integer is expected
      i -= '0';
      printf("\n");
      if ((i < 1) || (i > cnt))
      {
         printf("Invalid selection\n\n");
      }
      else
      {
         done = VI_TRUE;
      }

      printf("\nPlease select: ");
      while ((i = getchar()) == EOF);
      i -= '0';
      fflush(stdin);
      printf("\n");
      if ((i < 1) || (i > cnt))
      {
         printf("Invalid selection\n\n");
      }
      else
      {
         done = 1;
      }
   } while (!done);

   // Copy resource string to static buffer
   err = TLPM_getRsrcName(0, i - 1, rsrcDescr);

   return (err);
   */

}
