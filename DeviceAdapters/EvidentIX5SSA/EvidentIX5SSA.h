///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX5SSA.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX5-SSA XY Stage device adapter
//
// COPYRIGHT:     University of California, San Francisco, 2025
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
//
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include "DeviceBase.h"
#include "EvidentIX5SSAProtocol.h"
#include "EvidentIX5SSAModel.h"
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// Use protocol and model from separate files

//////////////////////////////////////////////////////////////////////////////
// EvidentIX5SSA - IX5-SSA XY Stage Controller
//////////////////////////////////////////////////////////////////////////////

class EvidentIX5SSA : public CXYStageBase<EvidentIX5SSA>
{
public:
   EvidentIX5SSA();
   ~EvidentIX5SSA();

   // MMDevice API
   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   bool SupportsDeviceDetection(void) { return true; }
   MM::DeviceDetectionStatus DetectDevice(void);

   // XYStage API
   int SetPositionSteps(long x, long y);
   int GetPositionSteps(long& x, long& y);
   int SetRelativePositionSteps(long x, long y);
   int Home();
   int Stop();
   int SetOrigin();
   int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
   int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
   double GetStepSizeXUm() { return stepSizeXUm_; }
   double GetStepSizeYUm() { return stepSizeYUm_; }
   int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

   // Action interface
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJogEnable(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJogSensitivity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJogDirectionX(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJogDirectionY(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   // Threading
   void MonitorThreadFunc();
   void StartMonitoring();
   void StopMonitoring();

   // Command/response pattern
   int ExecuteCommand(const std::string& cmd, std::string& response, long timeoutMs = 4000);
   int SendCommand(const std::string& cmd);
   int GetResponse(std::string& response, long timeoutMs = 4000);

   // Notification processing
   void ProcessNotification(const std::string& notification);

   // Helper methods
   int LoginToRemoteMode();
   int VerifyDevice();
   int QueryVersion();
   int InitializeStage();
   int EnableJog(bool enable);
   int EnableNotifications(bool enable);

   // State
   bool initialized_;
   std::string port_;
   std::string name_;
   double stepSizeXUm_;
   double stepSizeYUm_;

   // Model - thread-safe state storage
   IX5SSA::StageModel model_;

   // Threading
   std::thread monitorThread_;
   std::atomic<bool> stopMonitoring_;

   // Command synchronization
   mutable std::mutex commandMutex_;  // Protects command sending
   std::mutex responseMutex_;         // Protects response handling
   std::condition_variable responseCV_;
   std::string pendingResponse_;
   bool responseReady_;
};
