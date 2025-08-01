///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIXYStage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI XY Stage device adapter
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
// BASED ON:      ASIStage.h and others
//

#ifndef ASIXYSTAGE_H
#define ASIXYSTAGE_H

#include "ASIPeripheralBase.h"
#include "MMDevice.h"
#include "DeviceBase.h"

class CXYStage : public ASIPeripheralBase<CXYStageBase, CXYStage>
{
public:
   CXYStage(const char* name);
   ~CXYStage() { }
  
   // Device API
   // ----------
   int Initialize();
   bool Busy();

   // XYStage API
   // -----------
   int Stop();

   // XYStageBase uses these functions to move the stage
   // the step size is the programming unit for dimensions and is integer
   // see https://sourceforge.net/p/micro-manager/mailman/message/31318649/
   double GetStepSizeXUm() {return stepSizeYUm_;}
   double GetStepSizeYUm() {return stepSizeYUm_;}
   int GetPositionSteps(long& x, long& y);
   int SetPositionSteps(long x, long y);
   int SetRelativePositionSteps(long x, long y);
   int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
   int SetOrigin();
   int SetXOrigin();
   int SetYOrigin();
   int Home();
   int SetHome();
   int Move (double vx, double vy);

   int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = ttl_trigger_enabled_; return DEVICE_OK; }
   int GetXYStageSequenceMaxLength(long& nrEvents) const { nrEvents = ring_buffer_capacity_; return DEVICE_OK; }

   int StartXYStageSequence();
   int StopXYStageSequence();
   int ClearXYStageSequence();
   int AddToXYStageSequence(double positionX, double positionY);
   int SendXYStageSequence();

   // leave default implementation which call corresponding "Steps" functions
   //    while accounting for mirroring and so forth
//      int SetPositionUm(double x, double y);
//      int SetRelativePositionUm(double dx, double dy);
//      int SetAdapterOriginUm(double x, double y);
//      int GetPositionUm(double& x, double& y);

   // below aren't implemented yet
   int GetLimitsUm(double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/) { return DEVICE_UNSUPPORTED_COMMAND; }

   // action interface
   // ----------------
   int OnSaveCardSettings     (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRefreshProperties    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWaitTime             (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnNrExtraMoveReps      (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeedGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnSpeedXMicronsPerSec  (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeedYMicronsPerSec  (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeedX               (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSpeedGeneric(pProp, eAct, axisLetterX_); }
   int OnSpeedY               (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnSpeedGeneric(pProp, eAct, axisLetterY_); }
   int OnBacklashGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnBacklashX            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnBacklashGeneric(pProp, eAct, axisLetterX_); }
   int OnBacklashY            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnBacklashGeneric(pProp, eAct, axisLetterY_); }
   int OnDriftErrorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnDriftErrorX          (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDriftErrorGeneric(pProp, eAct, axisLetterX_); }
   int OnDriftErrorY          (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnDriftErrorGeneric(pProp, eAct, axisLetterY_); }
   int OnFinishErrorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnFinishErrorX         (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnFinishErrorGeneric(pProp, eAct, axisLetterX_); }
   int OnFinishErrorY         (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnFinishErrorGeneric(pProp, eAct, axisLetterY_); }
   int OnAccelerationGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnAccelerationX        (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnAccelerationGeneric(pProp, eAct, axisLetterX_); }
   int OnAccelerationY        (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnAccelerationGeneric(pProp, eAct, axisLetterY_); }
   int OnLowerLimGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnLowerLimX            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnLowerLimGeneric(pProp, eAct, axisLetterX_); }
   int OnLowerLimY            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnLowerLimGeneric(pProp, eAct, axisLetterY_); }
   int OnUpperLimGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnUpperLimX            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnUpperLimGeneric(pProp, eAct, axisLetterX_); }
   int OnUpperLimY            (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnUpperLimGeneric(pProp, eAct, axisLetterY_); }
   int OnMaintainStateGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnMaintainStateX       (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnMaintainStateGeneric(pProp, eAct, axisLetterX_); }
   int OnMaintainStateY       (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnMaintainStateGeneric(pProp, eAct, axisLetterY_); }
   int OnAdvancedProperties   (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnOvershoot            (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnKIntegral            (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnKProportional        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnKDerivative          (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnKDrive               (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnKFeedforward         (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAAlign               (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAZeroX               (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAZeroY               (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMotorControlGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnMotorControlX        (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnMotorControlGeneric(pProp, eAct, axisLetterX_); }
   int OnMotorControlY        (MM::PropertyBase* pProp, MM::ActionType eAct) { return OnMotorControlGeneric(pProp, eAct, axisLetterY_); }
   int OnJoystickFastSpeed    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystickSlowSpeed    (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystickMirror       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystickRotate       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystickEnableDisable(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWheelFastSpeed       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWheelSlowSpeed       (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWheelMirror          (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAxisPolarityX        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAxisPolarityY        (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanState            (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanFastAxis         (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanSlowAxis         (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanPattern          (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanFastStartPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanFastStopPosition (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanSlowStartPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanSlowStopPosition (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanNumLines         (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanSettlingTime     (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanOvershootDistance(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanRetraceSpeedPercent(MM::PropertyBase* pProp, MM::ActionType eAct);
   // ring buffer properties
   int OnRBDelayBetweenPoints (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRBMode               (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRBTrigger            (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRBRunning            (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnUseSequence          (MM::PropertyBase* pProp, MM::ActionType eAct);
   // vector properties (API only half-defined so far, would prefer to use API in long run)
   int OnVectorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, const std::string& axisLetter);
   int OnVectorX(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisLetterX_); }
   int OnVectorY(MM::PropertyBase* pProp, MM::ActionType eAct) { return OnVectorGeneric(pProp, eAct, axisLetterY_); }


private:
   double unitMultX_;
   double unitMultY_;
   double stepSizeXUm_;
   double stepSizeYUm_;
   std::string axisLetterX_;
   std::string axisLetterY_;
   bool advancedPropsEnabled_;
   bool speedTruth_;
   double lastSpeedX_;
   double lastSpeedY_;
   bool ring_buffer_supported_;
   long ring_buffer_capacity_;
   bool ttl_trigger_supported_;
   bool ttl_trigger_enabled_;
   std::vector<double> sequenceX_;
   std::vector<double> sequenceY_;

   int OnSaveJoystickSettings();
   int getMinMaxSpeed(const std::string& axisLetter, double& minSpeed, double& maxSpeed);
};

#endif // ASIXYSTAGE_H
