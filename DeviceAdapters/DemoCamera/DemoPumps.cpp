///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoPumps.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Demo pump device implementations (pressure and volumetric)
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

extern const char* g_PressurePumpDeviceName;
extern const char* g_VolumetricPumpDeviceName;
extern const char* g_PropImposedPressure;

///////////////////////////////////////////////////////////////////////////////
// DemoPressurePump implementation
///////////////////////////////////////////////////////////////////////////////

int DemoPressurePump::Initialize()
{
   CPropertyAction* pAct = new CPropertyAction(this, &DemoPressurePump::OnImposedPressure);
   int ret  = CreateFloatProperty(g_PropImposedPressure, 0.0, false, pAct);
   if (ret!= DEVICE_OK)
      return ret;
   SetPropertyLimits(g_PropImposedPressure, 0.0, 100.0);

   initialized_ = true;
   return DEVICE_OK;
}

int DemoPressurePump::OnImposedPressure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(currentPressure_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(currentPressure_);

      OnPropertyChanged(g_PropImposedPressure, CDeviceUtils::ConvertToString(currentPressure_));
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// DemoVolumetricPump implementation
// Note: All methods are implemented inline in the header file (DemoCamera.h)
///////////////////////////////////////////////////////////////////////////////
