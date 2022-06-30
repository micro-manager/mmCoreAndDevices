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
const char* g_On = "On";
const char* g_Off = "Off";


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

   if (result != DEVICE_OK)
      return;

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

   // get the calibration at the first memory position
   ViUInt16	memoryPosition = TLPM_INDEX_4;;
   ViChar 	sensorSerialNumber[TLPM_BUFFER_SIZE];
   ViChar 	calibrationDate[TLPM_BUFFER_SIZE];
   ViUInt16 calibrationPointsCount;
   ViChar 	author[TLPM_BUFFER_SIZE];
   ViUInt16 sensorPosition;
   err = TLPM_getPowerCalibrationPointsInformation(instrHdl_, memoryPosition, sensorSerialNumber, 
      calibrationDate, &calibrationPointsCount, author, &sensorPosition);
   if (err == VI_SUCCESS) {
      CreateStringProperty("Sensor Serial Number", sensorSerialNumber, true);
      CreateStringProperty("Calibration Date", calibrationDate, true);
      CreateStringProperty("Author", author, true);
   }

   CPropertyAction* pAct = new CPropertyAction(this, &PM100::OnValue);
   int ret = CreateProperty("Power", "0", MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &PM100::OnRawValue);
   ret = CreateStringProperty("RawPower", "0.0", true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &PM100::OnRawUnit);
   ret = CreateStringProperty("RawUnit", "W", true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &PM100::OnWavelength);
   ret = CreateFloatProperty("Wavelength", 488.0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   
   pAct = new CPropertyAction(this, &PM100::OnAutoRange);
   ret = CreateStringProperty("AutoRange", g_On, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("AutoRange", g_On);
   AddAllowedValue("AutoRange", g_Off);
   
   pAct = new CPropertyAction(this, &PM100::OnPowerRange);
   ret = CreateFloatProperty("PowerRange", 100.0, false, pAct);
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

int PM100::OnRawValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      ViStatus       err = VI_SUCCESS;
      ViReal64       power = 0.0;

      err = TLPM_measPower(instrHdl_, &power);
      if (err != VI_SUCCESS)
         return err;

      std::ostringstream os;
      os << std::setprecision(4) << power;
      pProp->Set(os.str().c_str());
   }
   return DEVICE_OK;
}

int PM100::OnRawUnit(MM::PropertyBase * pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      ViStatus       err = VI_SUCCESS;
      ViInt16        power_unit;

      err = TLPM_getPowerUnit(instrHdl_, &power_unit);
      std::string unit;
      switch (power_unit)
      {
         case TLPM_POWER_UNIT_DBM: unit = "dBm"; break;
         default: unit = "W"; break;
      }

      pProp->Set(unit.c_str());
   }
   return DEVICE_OK;
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


int PM100::OnWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ViStatus err = VI_SUCCESS;
   ViReal64 actWavelength;
   
   if (eAct == MM::BeforeGet)
   {

      err = TLPM_getWavelength(instrHdl_, TLPM_ATTR_SET_VAL, &actWavelength);
      if (err != VI_SUCCESS)
         return err;

      pProp->Set(actWavelength);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(actWavelength);
      return TLPM_setWavelength(instrHdl_,  actWavelength);
      
   }
   return DEVICE_OK;
}

int PM100::OnAutoRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ViStatus err = VI_SUCCESS;
   ViBoolean autoRangeMode = TLPM_AUTORANGE_POWER_OFF;
   std::string state = g_Off;

   if (eAct == MM::BeforeGet)
   {

      err = TLPM_getPowerAutorange(instrHdl_, &autoRangeMode);
      if (err != VI_SUCCESS)
         return err;
      if (autoRangeMode)
         state = g_On;

      pProp->Set(state.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(state);
      if (state == g_On)
         autoRangeMode = (ViBoolean) TLPM_AUTORANGE_POWER_ON;
      return TLPM_setPowerAutoRange(instrHdl_, autoRangeMode);

   }
   return DEVICE_OK;
}

int PM100::OnPowerRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ViStatus err = VI_SUCCESS;
   ViReal64 powerRange;

   if (eAct == MM::BeforeGet)
   {
      err = TLPM_getPowerRange(instrHdl_, TLPM_AUTORANGE_POWER_OFF, &powerRange);
      if (err != VI_SUCCESS)
         return err;

      pProp->Set(powerRange);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(powerRange);
      return TLPM_setPowerRange(instrHdl_, powerRange);

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