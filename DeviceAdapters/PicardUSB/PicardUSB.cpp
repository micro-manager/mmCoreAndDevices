// Picard Industries USB devices (PiUsbSDK)
//
// AUTHOR:        Mark A. Tsuchida
//
// COPYRIGHT:     2026 Board of Regents of the University of Wisconsin System
//
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

#include "PicardUSB.h"

#include "DeviceBase.h"
#include "ModuleInterface.h"

#include "PiUsb.h"

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

const char* g_SideFlipperName = "SideFlipper";
const char* g_GradientWheelName = "GradientWheel";
const char* g_PropSerialNumber = "Serial Number";
const char* g_PropPosition = "Position";
const char* g_PropState = "State";
const char* g_PropOpenPosition = "Open Position";
const char* g_OpenPositionExtended = "Extended";
const char* g_OpenPositionRetracted = "Retracted";

// SDK error n (1..9) maps to ERR_PI_BASE + n so each gets its own SetErrorText.
constexpr int ERR_PI_BASE = 100;         // 101..109 = PiUsb errors 1..9
constexpr int ERR_CONNECT_FAILED = 111;  // piConnect returned NULL without error code
constexpr int ERR_SERIAL_NOT_SET = 112;  // Serial Number property still 0

// Tweakables:
constexpr long flipTimeoutMs = 1000;
constexpr long gradientWheelMoveTimeoutMs = 5000;
constexpr int gradientWheelPositionToleranceCounts = 0;

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_SideFlipperName, MM::ShutterDevice, "Picard USB Side Flipper");
   RegisterDevice(g_GradientWheelName, MM::GenericDevice, "Picard USB Gradient Wheel");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == nullptr)
      return 0;
   if (std::strcmp(deviceName, g_SideFlipperName) == 0)
      return new SideFlipper();
   if (std::strcmp(deviceName, g_GradientWheelName) == 0)
      return new GradientWheel();
   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

namespace {

typedef int(__stdcall* FindFunc)(int*, int*, int);

std::vector<int> FindSerials(FindFunc find)
{
   int count = 0;
   int serials[256];
   int err = find(&count, serials, 256);
   if (err != PI_NO_ERROR)
      return std::vector<int>();
   return std::vector<int>(serials, serials + count);
}

int TranslatePiError(int piErrNo)
{
   if (piErrNo == PI_NO_ERROR)
      return DEVICE_OK;
   return ERR_PI_BASE + piErrNo;
}

std::vector<std::pair<int, std::string>> PiErrorTexts()
{
   return {
      { ERR_PI_BASE + 1, "PiUsb: device not found or disconnected" },
      { ERR_PI_BASE + 2, "PiUsb: device handle does not exist" },
      { ERR_PI_BASE + 3, "PiUsb: cannot create device object" },
      { ERR_PI_BASE + 4, "PiUsb: invalid device handle" },
      { ERR_PI_BASE + 5, "PiUsb: read timeout (device may be disconnected)" },
      { ERR_PI_BASE + 6, "PiUsb: read thread abandoned; reconnect the device" },
      { ERR_PI_BASE + 7, "PiUsb: read failed; reconnect the device" },
      { ERR_PI_BASE + 8, "PiUsb: invalid parameter" },
      { ERR_PI_BASE + 9, "PiUsb: write failed; reconnect the device" },
      { ERR_CONNECT_FAILED,
         "Could not connect to the device with the given serial number; "
         "check that it is connected and not in use" },
      { ERR_SERIAL_NOT_SET,
         "Serial Number must be set before initialization" },
   };
}

} // namespace

SideFlipper::SideFlipper()
{
   InitializeDefaultErrorMessages();
   for (const auto& e : PiErrorTexts())
      SetErrorText(e.first, e.second.c_str());

   CreateIntegerProperty(g_PropSerialNumber, 0, false,
      new CPropertyAction(this, &SideFlipper::OnSerialNumber), true);
   std::vector<int> serials = FindSerials(piFindFlippers);
   for (int s : serials)
      AddAllowedValue(g_PropSerialNumber, CDeviceUtils::ConvertToString(s));

   CreateStringProperty(g_PropOpenPosition, g_OpenPositionExtended, false,
      new CPropertyAction(this, &SideFlipper::OnOpenPosition), true);
   AddAllowedValue(g_PropOpenPosition, g_OpenPositionExtended);
   AddAllowedValue(g_PropOpenPosition, g_OpenPositionRetracted);
}

SideFlipper::~SideFlipper()
{
   Shutdown();
}

int SideFlipper::Initialize()
{
   if (handle_)
      return DEVICE_OK;

   if (serial_ <= 0)
      return ERR_SERIAL_NOT_SET;

   int e = 0;
   handle_ = piConnectFlipper(&e, static_cast<int>(serial_));
   if (!handle_)
      return e ? TranslatePiError(e) : ERR_CONNECT_FAILED;

   int state = PI_FLIPPER_RETRACTED;
   int err = piGetFlipperState(&state, handle_);
   if (err != PI_NO_ERROR)
   {
      piDisconnectFlipper(handle_);
      handle_ = nullptr;
      return TranslatePiError(err);
   }
   commandedState_ = state;

   CreateIntegerProperty(g_PropState, 0, false,
      new CPropertyAction(this, &SideFlipper::OnState));
   AddAllowedValue(g_PropState, "0");
   AddAllowedValue(g_PropState, "1");

   return DEVICE_OK;
}

int SideFlipper::Shutdown()
{
   if (handle_)
   {
      piDisconnectFlipper(handle_);
      handle_ = nullptr;
   }
   return DEVICE_OK;
}

void SideFlipper::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_SideFlipperName);
}

bool SideFlipper::Busy()
{
   if (!handle_)
      return false;
   // Bound by a settle timeout so a physically blocked flipper cannot hang MM.
   if ((GetCurrentMMTime() - lastCommandTime_).getMsec() > flipTimeoutMs)
      return false;
   // TODO: We probably want to add an extra wait after state changes
   int state;
   if (piGetFlipperState(&state, handle_) != PI_NO_ERROR)
      return false;
   return state != commandedState_;
}

int SideFlipper::SetOpen(bool open)
{
   int state = (open == openIsExtended_) ? PI_FLIPPER_EXTENDED : PI_FLIPPER_RETRACTED;
   int err = piSetFlipperState(state, handle_);
   if (err != PI_NO_ERROR)
      return TranslatePiError(err);
   commandedState_ = state;
   lastCommandTime_ = GetCurrentMMTime();
   return DEVICE_OK;
}

int SideFlipper::GetOpen(bool& open)
{
   int state;
   int err = piGetFlipperState(&state, handle_);
   if (err != PI_NO_ERROR)
      return TranslatePiError(err);
   open = ((state == PI_FLIPPER_EXTENDED) == openIsExtended_);
   return DEVICE_OK;
}

int SideFlipper::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(serial_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(serial_);
   }
   return DEVICE_OK;
}

int SideFlipper::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool open;
      int err = GetOpen(open);
      if (err != DEVICE_OK)
         return err;
      pProp->Set(open ? 1L : 0L);
   }
   else if (eAct == MM::AfterSet)
   {
      long v;
      pProp->Get(v);
      return SetOpen(v != 0);
   }
   return DEVICE_OK;
}

int SideFlipper::OnOpenPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(openIsExtended_ ? g_OpenPositionExtended : g_OpenPositionRetracted);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string v;
      pProp->Get(v);
      openIsExtended_ = (v == g_OpenPositionExtended);
   }
   return DEVICE_OK;
}

GradientWheel::GradientWheel()
{
   InitializeDefaultErrorMessages();
   for (const auto& e : PiErrorTexts())
      SetErrorText(e.first, e.second.c_str());

   CreateIntegerProperty(g_PropSerialNumber, 0, false,
      new CPropertyAction(this, &GradientWheel::OnSerialNumber), true);
   std::vector<int> serials = FindSerials(piFindGWheels);
   for (int s : serials)
   {
      AddAllowedValue(g_PropSerialNumber, CDeviceUtils::ConvertToString(s));
   }
}

GradientWheel::~GradientWheel()
{
   Shutdown();
}

int GradientWheel::Initialize()
{
   if (!handle_)
      return DEVICE_OK;

   if (serial_ <= 0)
      return ERR_SERIAL_NOT_SET;

   int e = 0;
   handle_ = piConnectGWheel(&e, static_cast<int>(serial_));
   if (!handle_)
      return e ? TranslatePiError(e) : ERR_CONNECT_FAILED;

   int pos = 1;
   int err = piGetGWheelPosition(&pos, handle_);
   if (err != PI_NO_ERROR)
   {
      piDisconnectGWheel(handle_);
      handle_ = nullptr;
      return TranslatePiError(err);
   }
   if (pos < 1)
      pos = 1;
   else if (pos > 1023)
      pos = 1023;
   targetPosition_ = pos;

   CreateIntegerProperty(g_PropPosition, targetPosition_, false,
      new CPropertyAction(this, &GradientWheel::OnPosition));
   SetPropertyLimits(g_PropPosition, 1, 1023);

   return DEVICE_OK;
}

int GradientWheel::Shutdown()
{
   if (handle_)
   {
      piDisconnectGWheel(handle_);
      handle_ = nullptr;
   }
   return DEVICE_OK;
}

void GradientWheel::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_GradientWheelName);
}

bool GradientWheel::Busy()
{
   if (!handle_)
      return false;
   // No moving-status API; compare readback to target, bounded by a timeout.
   if ((GetCurrentMMTime() - moveStartTime_).getMsec() > gradientWheelMoveTimeoutMs)
      return false;
   // TODO We might want to add an extra wait after within tolerance
   int pos;
   if (piGetGWheelPosition(&pos, handle_) != PI_NO_ERROR)
      return false;
   return std::abs(pos - targetPosition_) > gradientWheelPositionToleranceCounts;
}

int GradientWheel::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(serial_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(serial_);
   }
   return DEVICE_OK;
}

int GradientWheel::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int pos;
      int err = piGetGWheelPosition(&pos, handle_);
      if (err != PI_NO_ERROR)
         return TranslatePiError(err);
      pProp->Set(static_cast<long>(pos));
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      int err = piSetGWheelPosition(static_cast<int>(pos), handle_);
      if (err != PI_NO_ERROR)
         return TranslatePiError(err);
      targetPosition_ = static_cast<int>(pos);
      moveStartTime_ = GetCurrentMMTime();
   }
   return DEVICE_OK;
}
