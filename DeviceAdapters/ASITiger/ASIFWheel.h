///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIFWheel.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI filter wheel adapter for Tiger
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Jon Daniels (jon@asiimaging.com) 09/2013
//

#pragma once

#include "ASIPeripheralBase.h"
#include "MMDevice.h"
#include "DeviceBase.h"

class CFWheel : public ASIPeripheralBase<CStateDeviceBase, CFWheel>
{
public:
    explicit CFWheel(const char* name);
    ~CFWheel() = default;

   // Generic device API
   int Initialize();
   bool Busy();

   // State device API
   unsigned long GetNumberOfPositions() const { return numPositions_; }

   // action interface
   int OnSaveCardSettings(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRefreshProperties(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnState       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLabel       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpin        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVelocity    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeedSetting(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLockMode    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnOffset      (MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int SelectWheelOverride();
    int SelectWheel();
    void ForcePropertyRefresh();

    // currently selected wheel, shared among all instances of this class
    inline static std::string selectedWheel_ = g_EmptyAxisLetterStr;

    std::string wheelNumber_ = g_EmptyAxisLetterStr; // 0..9 for filter wheels instead of A..Z
    unsigned int numPositions_ = 0; // will read actual number of positions
    unsigned int curPosition_ = 0;  // will read actual position
    bool spinning_ = false;
};
