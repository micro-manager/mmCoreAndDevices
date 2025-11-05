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
#include <thread>
#include <atomic>
#include <mutex>
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
    int DiscoverDevices();
    bool IsDevicePresent(EvidentIX85::DeviceType type) const;

    // Notification control
    int EnableNotification(const char* cmd, bool enable);

private:
    // Initialization helpers
    int SetRemoteMode();
    int GetVersion(std::string& version);
    int GetUnit(std::string& unit);
    int ClearPort();

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

    // Monitoring thread
    void StartMonitoring();
    void StopMonitoring();
    void MonitorThreadFunc();

    // Member variables
    bool initialized_;
    std::string port_;
    long answerTimeoutMs_;
    EvidentIX85::MicroscopeModel model_;

    // Threading
    std::thread monitorThread_;
    std::atomic<bool> stopMonitoring_;
    mutable std::mutex commandMutex_;  // Protects serial communication

    // State
    std::string version_;
    std::string unit_;
    std::vector<EvidentIX85::DeviceType> availableDevices_;
};
