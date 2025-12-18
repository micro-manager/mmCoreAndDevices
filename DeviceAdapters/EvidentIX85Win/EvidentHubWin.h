///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentHubWin.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85Win microscope hub (SDK-based)
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
#include "EvidentModelWin.h"
#include "EvidentProtocolWin.h"
#include "EvidentIX85Win.h"
#include "EvidentSDK.h"
#include <windows.h>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>
#include <queue>
#include <future>
#include <thread>
#include <atomic>

// Command task structure for worker thread queue
struct CommandTask {
    std::string command;
    std::promise<std::pair<int, std::string>> responsePromise;

    CommandTask(std::string cmd) : command(std::move(cmd)) {}

    // Move-only type (promise is not copyable)
    CommandTask(CommandTask&&) = default;
    CommandTask& operator=(CommandTask&&) = default;
    CommandTask(const CommandTask&) = delete;
    CommandTask& operator=(const CommandTask&) = delete;
};

class EvidentHubWin : public HubBase<EvidentHubWin>
{
public:
    EvidentHubWin();
    ~EvidentHubWin();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    bool SupportsDeviceDetection(void) { return true; }
    MM::DeviceDetectionStatus DetectDevice(void);
    // Hub API
    int DetectInstalledDevices();

    // Action handlers
    int OnSerialPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDLLPath(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAnswerTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHandSwitchJog(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHandSwitchSwitches(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHandSwitchCondenser(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnHandSwitchIndicators(MM::PropertyBase* pProp, MM::ActionType eAct);

    // Hub interface for devices to access state
    EvidentIX85Win::MicroscopeModel* GetModel() { return &model_; }

    // Command execution (thread-safe with worker thread queue)
    std::future<std::pair<int, std::string>> ExecuteCommandAsync(const std::string& command);
    int ExecuteCommand(const std::string& command, std::string& response);
    int SendCommand(const std::string& command);
    int GetResponse(std::string& response, long timeoutMs = -1);

    // Device discovery
    bool IsDevicePresent(EvidentIX85Win::DeviceType type) const;
    std::string GetDeviceVersion(EvidentIX85Win::DeviceType type) const;

    // Objective lens information
    const std::vector<EvidentIX85Win::ObjectiveInfo>& GetObjectiveInfo() const { return objectiveInfo_; }

    // Notification control
    int EnableNotification(const char* cmd, bool enable);

    void RegisterDeviceAsUsed(EvidentIX85Win::DeviceType type, MM::Device* device) { usedDevices_[type] = device;};
    void UnRegisterDeviceAsUsed(EvidentIX85Win::DeviceType type) { usedDevices_.erase(type); };

    int UpdateMirrorUnitIndicator(int position, bool async);
    int UpdateLightPathIndicator(int position, bool async);
    int UpdateEPIShutter1Indicator(int state, bool async);

    // DIA brightness memory for logical shutter
    int GetRememberedDIABrightness() const { return rememberedDIABrightness_; }
    void SetRememberedDIABrightness(int brightness) { rememberedDIABrightness_ = brightness; }

    // Measured Z-offset notification (when autofocus measures the offset)
    void NotifyMeasuredZOffsetChanged(long offsetSteps);

    int SetFocusPositionSteps(long position);

private:
    // Initialization helpers
    int SetRemoteMode();
    int SetSettingMode(bool enable);
    int EnterSettingMode();
    int ExitSettingMode();
    int GetUnit(std::string& unit);
    int GetUnitDirect(std::string& unit);
    int ClearPort();
    int DoDeviceDetection();
    int QueryObjectiveInfo();

    // Worker thread for command queue
    void CommandWorkerThread();
    int ExecuteCommandInternal(const std::string& command, std::string& response);

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
    int QueryOffsetLens();

    // Manual Control Unit (MCU) helpers
    int UpdateNosepieceIndicator(int position);

    // SDK DLL management
    int LoadSDK();
    int UnloadSDK();
    int EnumerateAndOpenInterface();

    // Notification processing (called from SDK callbacks)
    void ProcessNotification(const std::string& message);
    bool IsNotificationTag(const std::string& message) const;

    // SDK Callback handlers (static - route to instance methods)
    static int CALLBACK CommandCallbackStatic(ULONG MsgId, ULONG wParam, ULONG lParam,
                                               PVOID pv, PVOID pContext, PVOID pCaller);
    static int CALLBACK NotifyCallbackStatic(ULONG MsgId, ULONG wParam, ULONG lParam,
                                              PVOID pv, PVOID pContext, PVOID pCaller);
    static int CALLBACK ErrorCallbackStatic(ULONG MsgId, ULONG wParam, ULONG lParam,
                                             PVOID pv, PVOID pContext, PVOID pCaller);

    // Instance callback handlers
    int OnCommandComplete(EvidentSDK::MDK_MSL_CMD* pCmd);
    int OnNotification(const char* notificationStr);
    int OnError(EvidentSDK::MDK_MSL_CMD* pCmd);

    // Member variables
    bool initialized_;
    std::string port_;
    std::string dllPath_;  // Configurable SDK DLL path
    long answerTimeoutMs_;
    EvidentIX85Win::MicroscopeModel model_;

    // SDK DLL handles
    HMODULE dllHandle_;
    void* interfaceHandle_;  // Opaque SDK interface handle

    // SDK function pointers
    EvidentSDK::fn_MSL_PM_Initialize pfnInitialize_;
    EvidentSDK::fn_MSL_PM_EnumInterface pfnEnumInterface_;
    EvidentSDK::fn_MSL_PM_GetInterfaceInfo pfnGetInterfaceInfo_;
    EvidentSDK::fn_MSL_PM_GetPortName pfnGetPortName_;
    EvidentSDK::fn_MSL_PM_OpenInterface pfnOpenInterface_;
    EvidentSDK::fn_MSL_PM_CloseInterface pfnCloseInterface_;
    EvidentSDK::fn_MSL_PM_SendCommand pfnSendCommand_;
    EvidentSDK::fn_MSL_PM_RegisterCallback pfnRegisterCallback_;

    // Command synchronization (keep pattern - callbacks signal instead of monitor thread)
    mutable std::mutex commandMutex_;  // Protects command sending (deprecated - worker thread serializes)
    std::mutex responseMutex_;         // Protects response handling
    std::condition_variable responseCV_;
    std::string pendingResponse_;
    bool responseReady_;
    EvidentSDK::MDK_MSL_CMD pendingCommand_;  // Must be member to stay valid for async callback

    // Command queue for worker thread
    std::queue<CommandTask> commandQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    std::thread workerThread_;
    std::atomic<bool> workerRunning_;

    // State
    std::string version_;
    std::string unit_;
    std::vector<EvidentIX85Win::DeviceType> availableDevices_;
    std::vector<std::string> detectedDevicesByName_;
    std::vector<EvidentIX85Win::ObjectiveInfo> objectiveInfo_;  // Objective lens info for positions 1-6

    // MCU switch state
    int rememberedDIABrightness_;  // Remembered brightness for DIA switch toggle

    // Child devices
    std::map<EvidentIX85Win::DeviceType, MM::Device*> usedDevices_;
};
