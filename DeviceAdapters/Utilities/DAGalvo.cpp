///////////////////////////////////////////////////////////////////////////////
// FILE:          DAGalvo.cpp
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

#include <chrono>
#include <thread>


extern const char* g_NoDevice;
extern const char* g_DeviceNameDAGalvoDevice;

DAGalvo::DAGalvo() :
   daXDevice_(g_NoDevice),
   daYDevice_(g_NoDevice),
   shutter_(g_NoDevice),
   pulseIntervalUs_(100000),
   initialized_(false)
{
}

DAGalvo::~DAGalvo()
{
   Shutdown();
}

int DAGalvo::Initialize()
{
   std::string propNameX = "DA for X";
   std::string propNameY = "DA for Y";
   CPropertyAction* pAct = new CPropertyAction(this, &DAGalvo::OnDAX);
   int ret = CreateStringProperty(propNameX.c_str(), daXDevice_.c_str(), false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   pAct = new CPropertyAction(this, &DAGalvo::OnDAY);
   ret = CreateStringProperty(propNameY.c_str(), daYDevice_.c_str(), false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   AddAllowedValue(propNameX.c_str(), g_NoDevice);
   AddAllowedValue(propNameY.c_str(), g_NoDevice);
   // Get labels of DA (SignalIO) devices
   std::vector<std::string> daDevices;
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         AddAllowedValue(propNameX.c_str(), deviceName);
         AddAllowedValue(propNameY.c_str(), deviceName);
      }
      else
         break;
   }

   pAct = new CPropertyAction(this, &DAGalvo::OnShutter);
   ret = CreateStringProperty("Shutter", shutter_.c_str(), false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   AddAllowedValue("Shutter", g_NoDevice);
   char shutterDevice[MM::MaxStrLength];
   deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::ShutterDevice, shutterDevice, deviceIterator++);
      if (0 < strlen(shutterDevice))
      {
         AddAllowedValue("Shutter", shutterDevice);
      }
      else
         break;
   }

   return DEVICE_OK;
}

int DAGalvo::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

void DAGalvo::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameDAGalvoDevice);
}

bool DAGalvo::Busy()
{
   MM::SignalIO* dax = static_cast<MM::SignalIO*>(GetDevice(daXDevice_.c_str()));
   if (dax && dax->Busy())
      return true;

   MM::SignalIO* day = static_cast<MM::SignalIO*>(GetDevice(daYDevice_.c_str()));
   if (day && day->Busy())
      return true;

   return false;
}

int DAGalvo::PointAndFire(double x, double y, double timeUs)
{
   int ret = SetPosition(x, y);
   if (ret != DEVICE_OK)
      return ret;
   MM::Shutter* s = static_cast<MM::Shutter*>(GetDevice(shutter_.c_str()));
   if (!s)
      return ERR_NO_SHUTTER_DEVICE_FOUND;

   // Should we do this non-blocking instead?
   ret = s->SetOpen(true);
   if (ret != DEVICE_OK)
      return ret;
   std::this_thread::sleep_for(std::chrono::microseconds((long long)timeUs));
   return s->SetOpen(false);

}

/*
* This appears to set the time a single spot should be illuminated
*/
int DAGalvo::SetSpotInterval(double pulseIntervalUs)
{
   pulseIntervalUs_ = pulseIntervalUs;
   return DEVICE_OK;
}

int DAGalvo::SetIlluminationState(bool on)
{
   MM::Shutter* s = static_cast<MM::Shutter*>(GetDevice(shutter_.c_str()));
   if (!s)
      return ERR_NO_SHUTTER_DEVICE_FOUND;
   return s->SetOpen(on);
}

int DAGalvo::SetPosition(double x, double y)
{
   MM::SignalIO* dax = static_cast<MM::SignalIO*>(GetDevice(daXDevice_.c_str()));
   if (!dax)
      return ERR_NO_DA_DEVICE_FOUND;
   int ret = dax->SetSignal(x);
   if (ret != DEVICE_OK)
      return ret;
   MM::SignalIO* day = static_cast<MM::SignalIO*>(GetDevice(daYDevice_.c_str()));
   if (!day)
      return ERR_NO_DA_DEVICE_FOUND;
   return day->SetSignal(y);
}

int DAGalvo::GetPosition(double& x, double& y)
{
   MM::SignalIO* dax = static_cast<MM::SignalIO*>(GetDevice(daXDevice_.c_str()));
   if (!dax)
      return ERR_NO_DA_DEVICE_FOUND;
   int ret = dax->GetSignal(x);
   if (ret != DEVICE_OK)
      return ret;
   MM::SignalIO* day = static_cast<MM::SignalIO*>(GetDevice(daYDevice_.c_str()));
   if (!day)
      return ERR_NO_DA_DEVICE_FOUND;
   return day->GetSignal(y);
}

double DAGalvo::GetXRange()
{
   MM::SignalIO* dax = static_cast<MM::SignalIO*>(GetDevice(daXDevice_.c_str()));
   if (!dax)
      return ERR_NO_DA_DEVICE_FOUND;
   double xMin;
   double xMax;
   dax->GetLimits(xMin, xMax);
   return xMax - xMin;
}

double DAGalvo::GetXMinimum() {
   MM::SignalIO* dax = static_cast<MM::SignalIO*>(GetDevice(daXDevice_.c_str()));
   if (!dax)
      return ERR_NO_DA_DEVICE_FOUND;
   double xMin;
   double xMax;
   dax->GetLimits(xMin, xMax);
   return xMin;
}

double DAGalvo::GetYRange()
{
   MM::SignalIO* day = static_cast<MM::SignalIO*>(GetDevice(daYDevice_.c_str()));
   if (!day)
      return ERR_NO_DA_DEVICE_FOUND;
   double yMin;
   double yMax;
   day->GetLimits(yMin, yMax);
   return yMax - yMin;
}

double DAGalvo::GetYMinimum()
{
   MM::SignalIO* day = static_cast<MM::SignalIO*>(GetDevice(daYDevice_.c_str()));
   if (!day)
      return ERR_NO_DA_DEVICE_FOUND;
   double yMin;
   double yMax;
   day->GetLimits(yMin, yMax);
   return yMin;
}

int DAGalvo::AddPolygonVertex(int /* polygonIndex */, double /* x */, double /* y */)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int DAGalvo::DeletePolygons()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int DAGalvo::RunSequence()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int DAGalvo::LoadPolygons()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

int DAGalvo::SetPolygonRepetitions(int repetitions)
{
   nrRepetitions_ = repetitions;

   return DEVICE_OK;
}

int DAGalvo::RunPolygons()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}
int DAGalvo::StopSequence()
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

// TODO: once we control illumination, this can be used to provide feedback
// Careful: the Galvo interface is not well documented.
int DAGalvo::GetChannel(char* /* channelName */)
{
   return DEVICE_OK;;
}

int DAGalvo::OnDAX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(daXDevice_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string daDeviceName;
      pProp->Get(daDeviceName);
      MM::SignalIO* dax = (MM::SignalIO*)GetDevice(daDeviceName.c_str());
      if (dax != 0) {
         daXDevice_ = daDeviceName;
      }
      else
         daXDevice_ = g_NoDevice;
   }
   return DEVICE_OK;
}

int DAGalvo::OnDAY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(daYDevice_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string daDeviceName;
      pProp->Get(daDeviceName);
      MM::SignalIO* day = (MM::SignalIO*)GetDevice(daDeviceName.c_str());
      if (day != 0) {
         daYDevice_ = daDeviceName;
      }
      else
         daYDevice_ = g_NoDevice;
   }
   return DEVICE_OK;
}

int DAGalvo::OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(shutter_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string shutter;
      pProp->Get(shutter);
      MM::Shutter* s = (MM::Shutter*)GetDevice(shutter.c_str());
      if (s != 0) {
         shutter_ = shutter;
      }
      else
         shutter_ = g_NoDevice;
   }
   return DEVICE_OK;
}
