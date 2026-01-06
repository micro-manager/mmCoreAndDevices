///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoMagnifier.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of the demo camera.
//                Simulates generic digital camera and associated automated
//                microscope devices and enables testing of the rest of the
//                system without the need to connect to the actual hardware.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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

#include "DemoCamera.h"
#include <sstream>

extern const char* g_MagnifierDeviceName;

///////////////////////////////////////////////////////////////////////////////
// CDemoMagnifier implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
DemoMagnifier::DemoMagnifier ()
{
   CPropertyAction* pAct = new CPropertyAction (this, &DemoMagnifier::OnHighMag);
   CreateFloatProperty("High Position Magnification", 1.6, false, pAct, true);

   pAct = new CPropertyAction (this, &DemoMagnifier::OnVariable);
   std::string propName = "Freely variable or fixed magnification";
   CreateStringProperty(propName.c_str(), "Fixed", false, pAct, true);
   AddAllowedValue(propName.c_str(), "Fixed");
   AddAllowedValue(propName.c_str(), "Variable");

   // parent ID display
   CreateHubIDProperty();
};

void DemoMagnifier::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_MagnifierDeviceName);
}

int DemoMagnifier::Initialize()
{
   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (pHub)
   {
      char hubLabel[MM::MaxStrLength];
      pHub->GetLabel(hubLabel);
      SetParentID(hubLabel); // for backward comp.
   }
   else
      LogMessage(NoHubError);

   if (variable_)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &DemoMagnifier::OnZoom);
      int ret = CreateFloatProperty("Zoom", zoomPosition_, false, pAct);
      if (ret != DEVICE_OK)
         return ret;
      SetPropertyLimits("Zoom", 0.1, highMag_);
   } else
   {
      CPropertyAction* pAct = new CPropertyAction (this, &DemoMagnifier::OnPosition);
      int ret = CreateStringProperty("Position", "1x", false, pAct);
      if (ret != DEVICE_OK)
         return ret;

      position_ = 0;

      AddAllowedValue("Position", "1x");
      AddAllowedValue("Position", highMagString().c_str());
   }

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

std::string DemoMagnifier::highMagString() {
   std::ostringstream os;
   os << highMag_ << "x";
   return os.str();
}

double DemoMagnifier::GetMagnification() {
   if (variable_)
   {
      return zoomPosition_;
   }
   else
   {
      if (position_ == 0)
         return 1.0;
      return highMag_;
   }
}

int DemoMagnifier::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      std::string pos;
      pProp->Get(pos);
      if (pos == "1x")
      {
         position_ = 0;
      }
      else {
         position_ = 1;
      }
      OnMagnifierChanged();
   }

   return DEVICE_OK;
}

int DemoMagnifier::OnZoom(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(zoomPosition_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(zoomPosition_);
      OnMagnifierChanged();
   }
   return DEVICE_OK;
}

int DemoMagnifier::OnHighMag(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(highMag_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(highMag_);
      ClearAllowedValues("Position");
      AddAllowedValue("Position", "1x");
      AddAllowedValue("Position", highMagString().c_str());
   }

   return DEVICE_OK;
}

int DemoMagnifier::OnVariable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string response = "Fixed";
      if (variable_)
         response = "Variable";
      pProp->Set(response.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string response;
      pProp->Get(response);
      if (response == "Fixed")
         variable_ = false;
      else
         variable_ = true;
   }
   return DEVICE_OK;
}
