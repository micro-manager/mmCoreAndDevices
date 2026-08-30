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

#pragma once

#include "DeviceBase.h"

#include "PiUsb.h"

class SideFlipper : public CShutterBase<SideFlipper>
{
public:
   SideFlipper();
   ~SideFlipper() override;

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   // Shutter API
   int SetOpen(bool open = true) override;
   int GetOpen(bool& open) override;
   int Fire(double /*deltaT*/) override { return DEVICE_UNSUPPORTED_COMMAND; }

private:
   // Action handlers
   int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnOpenPosition(MM::PropertyBase* pProp, MM::ActionType eAct);

   PIHANDLE handle_{};
   long serial_{};
   int commandedState_ = PI_FLIPPER_RETRACTED;
   bool openIsExtended_ = true;
   MM::MMTime lastCommandTime_;
};

class GradientWheel : public CGenericBase<GradientWheel>
{
public:
   GradientWheel();
   ~GradientWheel() override;

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

private:
   // Action handlers
   int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);

   PIHANDLE handle_{};
   long serial_{};
   int targetPosition_ = 1;
   MM::MMTime moveStartTime_;
};
