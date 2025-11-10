///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentHub.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope hub
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
#include "EvidentModel.h"
#include "EvidentProtocol.h"
#include "EvidentIX85.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>

class EvidentHub : public HubBase<EvidentHub>
{
public:
    EvidentHub();
    ~EvidentHub();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Hub API
    int DetectInstalledDevices();

    // Action handlers
    int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAnswerTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);

    // Hub interface for devices to access state
    EvidentIX85::MicroscopeModel* GetModel() { return &model_; }

    // Command execution (thread-safe)
    int ExecuteCommand(const std::string& command, std::string& response);
    int SendCommand(const std::string& command);
    int GetResponse(std::string& response, long timeoutMs = -1);

    // Device discovery
    bool IsDevicePresent(EvidentIX85::DeviceType type) const;
    std::string GetDeviceVersion(EvidentIX85::DeviceType type) const;

    // Notification control
    int EnableNotification(const char* cmd, bool enable);

    void RegisterDeviceAsUsed(EvidentIX85::DeviceType type, MM::Device* device) { usedDevices_[type] = device;};
    void UnRegisterDeviceAsUsed(EvidentIX85::DeviceType type) { usedDevices_.erase(type); };

    int UpdateMirrorUnitIndicator(int position);
    int UpdateLightPathIndicator(int position);
    int UpdateEPIShutter1Indicator(int state);
    int UpdateDIABrightnessIndicator(int brightness);

    // DIA brightness memory for logical shutter
    int GetRememberedDIABrightness() const { return rememberedDIABrightness_; }
    void SetRememberedDIABrightness(int brightness) { rememberedDIABrightness_ = brightness; }

private:
    // Initialization helpers
    int SetRemoteMode();
    int GetVersion(std::string& version);
    int GetUnit(std::string& unit);
    int ClearPort();
    int DoDeviceDetection();
    int QueryDevicePresenceByVersion(int unitNumber, std::string& version);

    // Device query helpers
    int QueryFocus();
    int QueryNosepiece();
    int QueryMagnification();
    int QueryLightPath();
    int QueryCondenserTurret();
    int QueryDIAAperture();
    int QueryDIAShutter();
    int QueryPolarizer();
    int QueryDICPrism();
    int QueryDICRetardation();
    int QueryEPIShutter1();
    int QueryEPIShutter2();
    int QueryMirrorUnit1();
    int QueryMirrorUnit2();
    int QueryEPIND();
    int QueryRightPort();
    int QueryCorrectionCollar();
    int QueryManualControl();

    // Manual Control Unit (MCU) helpers
    int UpdateNosepieceIndicator(int position);

    // Monitoring thread
    void StartMonitoring();
    void StopMonitoring();
    void MonitorThreadFunc();
    void ProcessNotification(const std::string& message);
    bool IsNotificationTag(const std::string& message) const;

    // Member variables
    bool initialized_;
    std::string port_;
    long answerTimeoutMs_;
    EvidentIX85::MicroscopeModel model_;

    // Threading
    std::thread monitorThread_;
    std::atomic<bool> stopMonitoring_;
    mutable std::mutex commandMutex_;  // Protects command sending only

    // Response handling (monitoring thread passes responses to command thread)
    std::mutex responseMutex_;
    std::condition_variable responseCV_;
    std::string pendingResponse_;
    bool responseReady_;

    // State
    std::string version_;
    std::string unit_;
    std::vector<EvidentIX85::DeviceType> availableDevices_;
    std::vector<std::string> detectedDevicesByName_;

    // MCU switch state
    int rememberedDIABrightness_;  // Remembered brightness for DIA switch toggle

    // Child devices
    std::map<EvidentIX85::DeviceType, MM::Device*> usedDevices_;
};
