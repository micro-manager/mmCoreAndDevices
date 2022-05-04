///////////////////////////////////////////////////////////////////////////////
// FILE:          DAMonochromator.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                DAXYStage by Ed Simmon, 11/28/2011
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
//                2015-2016, Open Imaging, Inc.
//                Altos Labs, 2022
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
//

#ifdef _WIN32
// Prevent windows.h from defining min and max macros,
// which clash with std::min and std::max.
#define NOMINMAX
#endif

#include "Utilities.h"

extern const char* g_DeviceNameDAMonochromator;
extern const char* g_NoDevice;


DAMonochromator::DAMonochromator() :
   DADeviceName_(""),
   initialized_(false),
   open_(false),
   minVoltage_(0.0),
   maxVoltage_(10.0),
   minWavelength_(200.0),
   maxWavelength_(1000.0),
   openWavelength_(400.0),
   closedWavelength_(200.0),
   openVoltage_(4.0),
   closedVoltage_(0.0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid DA device");
   SetErrorText(ERR_NO_DA_DEVICE, "No DA Device selected");
   SetErrorText(ERR_NO_DA_DEVICE_FOUND, "No DA Device loaded");

   // Name
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameDAMonochromator, MM::String, true);

   // Description
   CreateProperty(MM::g_Keyword_Description, "DA device used to control a monochromator", MM::String, true);

   // minimum wavelength
   CPropertyAction* pAct = new CPropertyAction(this, &DAMonochromator::OnMinWavelength);
   CreateProperty("Minimum wavelength", "", MM::Float, false, pAct, true);

   // maximum wavelength
   pAct = new CPropertyAction(this, &DAMonochromator::OnMaxWavelength);
   CreateProperty("Maximum wavelength", "", MM::Float, false, pAct, true);

   // minimum voltage
   pAct = new CPropertyAction(this, &DAMonochromator::OnMinVoltage);
   CreateProperty("Minimum voltage", "", MM::Float, false, pAct, true);

   // maximum voltage
   pAct = new CPropertyAction(this, &DAMonochromator::OnMaxVoltage);
   CreateProperty("Maximum voltage", "", MM::Float, false, pAct, true);

   // off-state wavelength
   pAct = new CPropertyAction(this, &DAMonochromator::OnClosedWavelength);
   CreateProperty("Shutter Closed Wavelength", "", MM::Float, false, pAct, true);
}

DAMonochromator::~DAMonochromator()
{
   Shutdown();
}

void DAMonochromator::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameDAMonochromator);
}

int DAMonochromator::Initialize()
{
   // get list with available DA devices.
   // TODO: this is a initialization parameter, which makes it harder for the end-user to set up!
   availableDAs_.clear();
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableDAs_.push_back(std::string(deviceName));
      }
      else
         break;
   }


   CPropertyAction* pAct = new CPropertyAction(this, &DAMonochromator::OnDADevice);
   std::string defaultDA = "Undefined";
   if (availableDAs_.size() >= 1)
      defaultDA = availableDAs_[0];
   CreateProperty("DA Device", defaultDA.c_str(), MM::String, false, pAct, false);
   if (availableDAs_.size() >= 1)
      SetAllowedValues("DA Device", availableDAs_);
   else
      return ERR_NO_DA_DEVICE_FOUND;

   // This is needed, otherwise DeviceDA_ is not always set resulting in crashes
   // This could lead to strange problems if multiple DA devices are loaded
   SetProperty("DA Device", defaultDA.c_str());

   pAct = new CPropertyAction(this, &DAMonochromator::OnState);
   CreateProperty("State", "0", MM::Integer, false, pAct);
   AddAllowedValue("State", "0");
   AddAllowedValue("State", "1");

   pAct = new CPropertyAction(this, &DAMonochromator::OnOpenWavelength);
   CreateProperty("Wavelength", "0", MM::Float, false, pAct);
   SetPropertyLimits("Wavelength", minWavelength_, maxWavelength_);

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

bool DAMonochromator::Busy()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());

   if (da != 0)
      return da->Busy();

   // If we are here, there is a problem.  No way to report it.
   return false;
}

/*
 * Opens or closes the shutter.  Remembers voltage from the 'open' position
 */
int DAMonochromator::SetOpen(bool open)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());

   if (da == 0)
      return ERR_NO_DA_DEVICE;
   int ret = DEVICE_ERR;
   double voltage = closedVoltage_;
   if (open) voltage = openVoltage_;

   ret = da->SetSignal(voltage);
   if (ret == DEVICE_OK) open_ = open;

   return ret;
}

int DAMonochromator::GetOpen(bool& open)
{
   open = open_;
   return DEVICE_OK;
}

///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int DAMonochromator::OnDADevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DADeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      // Make sure that the "old" DA device is open:
      SetOpen(true);

      std::string DADeviceName;
      pProp->Get(DADeviceName);
      MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName.c_str());
      if (da != 0) {
         DADeviceName_ = DADeviceName;
      }
      else
         return ERR_INVALID_DEVICE_NAME;

      // Gates are open by default.  Start with shutter closed:
      SetOpen(false);
   }
   return DEVICE_OK;
}


int DAMonochromator::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool open;
      int ret = GetOpen(open);
      if (ret != DEVICE_OK)
         return ret;
      long state = 0;
      if (open)
         state = 1;
      pProp->Set(state);
   }
   else if (eAct == MM::AfterSet)
   {
      long state;
      pProp->Get(state);
      bool open = false;
      if (state == 1)
         open = true;
      return SetOpen(open);
   }
   return DEVICE_OK;
}

int DAMonochromator::OnOpenWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(openWavelength_);
   }
   else if (eAct == MM::AfterSet)
   {
      MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
      if (da == 0)
         return ERR_NO_DA_DEVICE;

      double val;
      pProp->Get(val);
      openWavelength_ = val;

      double volt = (openWavelength_ - minWavelength_) * (maxVoltage_ - minVoltage_) / (maxWavelength_ - minWavelength_) + minVoltage_;
      if (volt > maxVoltage_ || volt < minVoltage_)
         return ERR_POS_OUT_OF_RANGE;

      openVoltage_ = volt;

      if (open_) {
         int ret = da->SetSignal(openVoltage_);
         if (ret != DEVICE_OK) return ret;
      }
   }
   return DEVICE_OK;
}
int DAMonochromator::OnClosedWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(closedWavelength_);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      closedWavelength_ = val;

      double volt = (closedWavelength_ - minWavelength_) * (maxVoltage_ - minVoltage_) / (maxWavelength_ - minWavelength_) + minVoltage_;
      if (volt > maxVoltage_ || volt < minVoltage_)
         return ERR_POS_OUT_OF_RANGE;

      closedVoltage_ = volt;

   }
   return DEVICE_OK;
}

int DAMonochromator::OnMinWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minWavelength_);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      minWavelength_ = val;
   }
   return DEVICE_OK;
}
int DAMonochromator::OnMaxWavelength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxWavelength_);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      maxWavelength_ = val;
   }
   return DEVICE_OK;
}
int DAMonochromator::OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minVoltage_);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      minVoltage_ = val;
   }
   return DEVICE_OK;
}
int DAMonochromator::OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxVoltage_);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      maxVoltage_ = val;
   }
   return DEVICE_OK;
}

