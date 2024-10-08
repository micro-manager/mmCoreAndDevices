///////////////////////////////////////////////////////////////////////////////
// FILE:          ASILED.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI programmable logic card device adapter
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
// AUTHOR:        Jon Daniels (jon@asiimaging.com) 05/2014
//
// BASED ON:      ASIStage.h and others
//

#ifndef ASIPLOGIC_H
#define ASIPLOGIC_H

#include "ASIPeripheralBase.h"
#include "MMDevice.h"
#include "DeviceBase.h"

class CPLogic : public ASIPeripheralBase<CShutterBase, CPLogic>
{
public:
   CPLogic(const char* name);
   ~CPLogic() { }
  
   // Device API
   int Initialize();
   bool Busy() { return false; }

   // Shutter API
   int SetOpen(bool open = true);
   int GetOpen(bool& open);
   int Fire(double /*deltaT*/) { return DEVICE_UNSUPPORTED_COMMAND; }

   // action interface
   int OnPLogicMode           (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSetShutterChannel    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPLogicOutputState    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPLogicOutputStateUpper(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFrontpanelOutputState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBackplaneOutputState (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerSource        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnClearAllCellStates   (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSetCardPreset        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPointerPosition      (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellType         (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellConfig       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellInput1       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellInput2       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellInput3       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellInput4       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnEditCellUpdates      (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSaveCardSettings     (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRefreshProperties    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAdvancedProperties   (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCellType             (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnCellConfig           (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnInput1               (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnInput2               (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnInput3               (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnInput4               (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnIOType               (MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnIOSourceAddress      (MM::PropertyBase* pProp, MM::ActionType eAct, long index);


private:
   std::string axisLetter_;
   unsigned int numCells_;
   unsigned int currentPosition_;  // cached value of current position
   
   // PLogic Mode Table
   // --------------------
   // diSPIM | 4ch | 7ch | (useAsdiSPIMShutter_, useAs4ChShutter_, useAs7ChShutter_)
   // --------------------
   //     0  |  0  |  0  | "None"
   //     1  |  1  |  0  | "diSPIM Shutter"
   //     0  |  1  |  0  | "Four-channel shutter"
   //     0  |  0  |  1  | "Seven-channel shutter"
   //     1  |  0  |  1  | "Seven-channel TTL shutter"
   bool useAsdiSPIMShutter_;
   bool useAs4ChShutter_;
   bool useAs7ChShutter_;
   // used together with either useAs4ChShutter_ or useAs7ChShutter_,
   // takes into account address 41 backplane (TTL0 for CameraA)

   bool shutterOpen_;
   bool editCellUpdates_;
   bool advancedPropsEnabled_; // flag to only create advanced properties once

   int SetPositionDirectly(unsigned int position);
   int GetCellPropertyName(long index, std::string suffix, char* name);
   int GetIOPropertyName(long index, std::string suffix, char* name);
   int RefreshAdvancedCellPropertyValues(long index);
   int RefreshCurrentPosition();
   int RefreshEditCellPropertyValues();
};

#endif // ASIPLOGIC_H
