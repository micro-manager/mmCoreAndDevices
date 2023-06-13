// Micro-Manager Device Adapter for Backer & Hickl DCC/DCU
// Author: Mark A. Tsuchida
//
// Copyright 2023 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#pragma once

#include "DCCDCUConfig.h"

#include <algorithm>
#include <bitset>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Wrap DCC or DCU device control as RAII objects.

// The BH API is structured such that all DCC modules (or all DCU modules) are
// initialized and closed together (selection between hardware and simulation
// is also for all modules simultaneously).
// In order to wrap the category (DCC or DCU) as a HubDevice and each module as
// its peripheral, it is conveneint to also have an object for each module.

// Generate a temporary minimal .ini file
template <DCCOrDCU Model> class DCCDCUIniFile {
   using Config = DCCDCUConfig<Model>;

   std::string fileName_;

 public:
   static constexpr std::size_t MaxNrModules = Config::MaxNrModules;

   explicit DCCDCUIniFile(std::bitset<MaxNrModules> moduleSet, bool simulate)
       : fileName_([] {
            std::array<char, L_tmpnam> name;
            if (std::tmpnam(name.data()) == nullptr)
               return std::string();
            return std::string(name.data());
         }()) {
      if (fileName_.empty()) {
         return;
      }

      std::ofstream file(fileName_);
      if (not file) {
         fileName_.clear();
         return;
      }

      // Undocumented: The first line must be a comment starting with
      // "DCC100" or "DCCUSB"
      file << "; " << Config::IniHeaderComment << "\n\n";
      file << "[" << Config::IniBaseTag << "]\n\n";
      file << Config::IniSimulationKey << " = "
           << std::to_string(simulate ? Config::SimulationMode : 0) << "\n\n";
      for (int i = 0; i < MaxNrModules; ++i) {
         file << "[" << Config::IniModuleTagPrefix << std::to_string(i + 1)
              << "]\n\n";
         file << "active = " << (moduleSet[i] ? "1" : "0") << "\n\n";
      }
   }

   ~DCCDCUIniFile() {
      if (not fileName_.empty()) {
         std::remove(fileName_.c_str());
      }
   }

   auto FileName() const -> std::string { return fileName_; }
};

template <DCCOrDCU Model> class DCCDCUInterface;

// Used with std::shared_ptr. All interaction with hardware goes via the parent
// DCCDCUInterface, so that it can be synchronized with the polling thread.
template <DCCOrDCU Model> class DCCDCUModule {
   using ParentType = DCCDCUInterface<Model>;
   using Config = DCCDCUConfig<Model>;
   std::shared_ptr<ParentType> parent_;
   short moduleNo_;

 public:
   explicit DCCDCUModule(std::shared_ptr<ParentType> parent, short moduleNo)
       : parent_(parent), moduleNo_(moduleNo) {}

   void Close() {
      parent_->ClearChangeHandlers(moduleNo_);
      parent_->CloseModule(moduleNo_);
   }

   auto IsActive() const -> bool { return parent_->IsModuleActive(moduleNo_); }

   auto InitStatus(short& error) const -> short {
      return parent_->GetModuleInitStatus(moduleNo_, error);
   }

   auto ModInfo(short& error) const {
      return parent_->GetModuleInfo(moduleNo_, error);
   }

   auto GetConnectorParameterBool(short connNo, ConnectorFeature feature,
                                  short& error) -> bool {
      return parent_->GetConnectorParameter(moduleNo_, connNo, feature,
                                            error) != 0.0f;
   }

   auto GetConnectorParameterUInt(short connNo, ConnectorFeature feature,
                                  short& error) -> unsigned {
      return static_cast<unsigned>(
          parent_->GetConnectorParameter(moduleNo_, connNo, feature, error));
   }

   auto GetConnectorParameterFloat(short connNo, ConnectorFeature feature,
                                   short& error) -> float {
      return parent_->GetConnectorParameter(moduleNo_, connNo, feature, error);
   }

   void SetConnectorParameterBool(short connNo, ConnectorFeature feature,
                                  bool value, short& error) {
      parent_->SetConnectorParameter(moduleNo_, connNo, feature,
                                     (value ? 1.0f : 0.0f), error);
   }

   void SetConnectorParameterUInt(short connNo, ConnectorFeature feature,
                                  unsigned value, short& error) {
      parent_->SetConnectorParameter(moduleNo_, connNo, feature,
                                     static_cast<float>(value), error);
   }

   void SetConnectorParameterFloat(short connNo, ConnectorFeature feature,
                                   float value, short& error) {
      parent_->SetConnectorParameter(moduleNo_, connNo, feature, value, error);
   }

   auto GetGainHVLimit(short connNo, short& error) -> float {
      return parent_->GetGainHVLimit(moduleNo_, connNo, error);
   }

   void EnableAllOutputs(bool enable, short& error) {
      parent_->EnableAllOutputs(moduleNo_, enable, error);
   }

   void EnableConnectorOutputs(short connNo, bool enable, short& error) {
      parent_->EnableConnectorOutputs(moduleNo_, connNo, enable, error);
   }

   void ClearAllOverloads(short& error) {
      parent_->ClearAllOverloads(moduleNo_, error);
   }

   void ClearConnectorOverload(short connNo, short& error) {
      parent_->ClearConnectorOverload(moduleNo_, connNo, error);
   }

   auto IsOverloaded(short connNo, short& error) -> bool {
      return parent_->IsOverloaded(moduleNo_, connNo, error);
   }

   auto IsCoolerCurrentLimitReached(short connNo, short& error) -> bool {
      return parent_->IsCoolerCurrentLimitReached(moduleNo_, connNo, error);
   }

   void SetOverloadChangeHandler(short connNo,
                                 std::function<void(bool)> func) {
      parent_->SetOverloadChangeHandler(moduleNo_, connNo, func);
   }

   void SetCurrLmtChangeHandler(short connNo, std::function<void(bool)> func) {
      parent_->SetCurrLmtChangeHandler(moduleNo_, connNo, func);
   }
};

// Guard all interaction with DCC/DCU API. This mutex is shared between
// DCCDCUInterface<DCC> and DCCDCUInterface<DCU>, since there is no thread
// safety guarantee whatsoever by the BH DCC DLL.
extern std::mutex apiMutex;

// Used with std::shared_ptr.
// Because the destructor calls DCC(U)_close(), creating more than one instance
// for the same model will lead to crashes.
template <DCCOrDCU Model>
class DCCDCUInterface
    : public std::enable_shared_from_this<DCCDCUInterface<Model>> {
   using Config = DCCDCUConfig<Model>;

   bool failedBeforeInit_ = false;
   short initError_ = 0;

   // Module objects are created lazily after construction because they need
   // access to shared_from_this().
   std::array<std::shared_ptr<DCCDCUModule<Model>>, Config::MaxNrModules>
       modules_;
   bool modulesCreated_ = false;

   std::thread pollingThread_;
   std::mutex pollingMutex_; // Not to be held at the same time as apiMutex
   std::condition_variable pollingCondVar_;
   bool stopPollingRequested_ = false;

   // We allow at most one change handler per (module, connector).
   struct ModulePollingInfo {
      short prevOverload = 0;
      short prevCurrLmt = 0;
      std::array<std::function<void(bool)>, Config::NrConnectors>
          overloadChanged;
      std::array<std::function<void(bool)>, Config::NrConnectors>
          currLmtChanged;
   };
   std::array<ModulePollingInfo, Config::MaxNrModules> pollingInfo_;

   void CreateModules() {
      auto shared_me = this->shared_from_this();
      for (short i = 0; i < MaxNrModules; ++i) {
         modules_[i] = std::make_shared<DCCDCUModule<Model>>(shared_me, i);
      }
      modulesCreated_ = true;
   }

 public:
   static constexpr std::size_t MaxNrModules = Config::MaxNrModules;

   explicit DCCDCUInterface(std::bitset<MaxNrModules> moduleSet,
                            bool simulate) {
      auto iniFile = DCCDCUIniFile<Model>(moduleSet, simulate);
      auto iniFileName = iniFile.FileName();
      if (iniFileName.empty()) {
         failedBeforeInit_ = true;
      } else {
         std::lock_guard<std::mutex> lock(apiMutex);
         initError_ = Config::Init(iniFileName.c_str());
      }

      // We do not stop if there is an init error, because some modules might
      // have initialized successfully (not entirely clear from documentation).

      std::bitset<MaxNrModules> modulesToPoll;
      {
         std::lock_guard<std::mutex> lock(apiMutex);
         for (short i = 0; i < MaxNrModules; ++i) {
            if (Config::TestIfActive(i)) {
               modulesToPoll.set(i);
            }
         }
      }
      StartPollingThread(modulesToPoll);
   }

   ~DCCDCUInterface() {
      if (std::any_of(modules_.begin(), modules_.end(),
                      [](auto pm) { return !!pm; })) {
         assert(false); // Modules must be closed first.
      }

      StopAndJoinPollingThread();

      std::lock_guard<std::mutex> lock(apiMutex);
      (void)Config::Close();
   }

   auto IsSimulating() -> bool {
      std::lock_guard<std::mutex> lock(apiMutex);
      return Config::GetMode() != 0;
   }

   auto PreInitError() const -> bool { return failedBeforeInit_; }

   auto InitError() const -> short { return initError_; }

   auto GetModule(int moduleNo) -> std::shared_ptr<DCCDCUModule<Model>> {
      if (not modulesCreated_) {
         // Done lazily because we cannot call shared_from_this() in
         // constructor.
         CreateModules();
      }
      return modules_[moduleNo];
   }

   void CloseModule(short moduleNo) { modules_[moduleNo].reset(); }

   void CloseAllModules() {
      for (short i = 0; i < MaxNrModules; ++i) {
         modules_[i].reset();
      }
   }

   auto IsModuleActive(short moduleNo) -> bool {
      std::lock_guard<std::mutex> lock(apiMutex);
      return Config::TestIfActive(moduleNo) != 0;
   }

   auto GetModuleInitStatus(short moduleNo, short& error) -> short {
      std::lock_guard<std::mutex> lock(apiMutex);
      short ret{};
      error = Config::GetInitStatus(moduleNo, &ret);
      return ret;
   }

   auto GetModuleInfo(short moduleNo, short& error) ->
       typename Config::ModInfoType {
      std::lock_guard<std::mutex> lock(apiMutex);
      typename Config::ModInfoType ret{};
      error = Config::GetModuleInfo(moduleNo, &ret);
      return ret;
   }

   auto GetConnectorParameter(short moduleNo, short connNo,
                              ConnectorFeature feature, short& error)
       -> float {
      std::lock_guard<std::mutex> lock(apiMutex);
      float ret{};
      error = Config::GetParameter(
          moduleNo, Config::ConnectorParameterId(connNo, feature), &ret);
      return ret;
   }

   void SetConnectorParameter(short moduleNo, short connNo,
                              ConnectorFeature feature, float value,
                              short& error) {
      std::lock_guard<std::mutex> lock(apiMutex);
      error = Config::SetParameter(
          moduleNo, Config::ConnectorParameterId(connNo, feature), true,
          value);
   }

   auto GetGainHVLimit(short moduleNo, short connNo, short& error) -> float {
      std::lock_guard<std::mutex> lock(apiMutex);
      short shortLimit{};
      error = Config::GetGainHVLimit(moduleNo, connNo, &shortLimit);
      return static_cast<float>(shortLimit);
   }

   void EnableAllOutputs(short moduleNo, bool enable, short& error) {
      std::lock_guard<std::mutex> lock(apiMutex);
      error = Config::EnableAllOutputs(moduleNo, enable ? 1 : 0);
   }

   void EnableConnectorOutputs(short moduleNo, short connNo, bool enable,
                               short& error) {
      std::lock_guard<std::mutex> lock(apiMutex);
      error = Config::EnableOutput(moduleNo, connNo, enable ? 1 : 0);
   }

   void ClearAllOverloads(short moduleNo, short& error) {
      {
         std::lock_guard<std::mutex> lock(apiMutex);
         error = Config::ClearAllOverloads(moduleNo);
      }
      PollSoon();
   }

   void ClearConnectorOverload(short moduleNo, short connNo, short& error) {
      {
         std::lock_guard<std::mutex> lock(apiMutex);
         error = Config::ClearOverload(moduleNo, connNo);
      }
      PollSoon();
   }

   auto IsOverloaded(short moduleNo, short connNo, short& error) -> bool {
      std::lock_guard<std::mutex> lock(apiMutex);
      short state{};
      error = Config::GetOverloadState(moduleNo, &state);
      return state & (1 << connNo);
   }

   auto IsCoolerCurrentLimitReached(short moduleNo, short connNo, short& error)
       -> bool {
      std::lock_guard<std::mutex> lock(apiMutex);
      short state{};
      error = Config::GetCurrLmtState(moduleNo, &state);
      return state & (1 << connNo);
   }

   // 'func' must not set change handlers.
   void SetOverloadChangeHandler(short moduleNo, short connNo,
                                 std::function<void(bool)> func) {
      std::lock_guard<std::mutex> lock(pollingMutex_);
      pollingInfo_[moduleNo].overloadChanged[connNo] = func;
   }

   // 'func' must not set change handlers.
   void SetCurrLmtChangeHandler(short moduleNo, short connNo,
                                std::function<void(bool)> func) {
      std::lock_guard<std::mutex> lock(pollingMutex_);
      pollingInfo_[moduleNo].currLmtChanged[connNo] = func;
   }

   void ClearChangeHandlers(short moduleNo) {
      std::lock_guard<std::mutex> lock(pollingMutex_);
      for (short i = 0; i < Config::NrConnectors; ++i) {
         pollingInfo_[moduleNo].overloadChanged[i] = {};
         pollingInfo_[moduleNo].currLmtChanged[i] = {};
      }
   }

   void PollSoon() {
      // Interrupt polling thread's sleep so that changes will be sent soon.
      pollingCondVar_.notify_one();
   }

 private:
   void StartPollingThread(std::bitset<MaxNrModules> modulesToPoll) {
      pollingThread_ = std::thread([this, modulesToPoll]() {
         std::unique_lock<std::mutex> pollingLock(pollingMutex_);
         for (;;) {
            if (stopPollingRequested_) {
               break;
            }
            pollingCondVar_.wait_for(pollingLock, std::chrono::seconds(1));
            if (stopPollingRequested_) {
               break;
            }

            pollingLock.unlock();

            std::array<short, MaxNrModules> overload;
            std::array<short, MaxNrModules> currLmt;
            {
               std::lock_guard<std::mutex> lock(apiMutex);
               for (short i = 0; i < MaxNrModules; ++i) {
                  if (not modulesToPoll.test(i)) {
                     overload[i] = currLmt[i] = 0;
                     continue;
                  }
                  short err = Config::GetOverloadState(i, &overload[i]);
                  if (err) {
                     break; // Give up on polling
                  }
                  err = Config::GetCurrLmtState(i, &currLmt[i]);
                  if (err) {
                     break; // Give up on polling
                  }
               }
            }

            pollingLock.lock();

            for (short i = 0; i < MaxNrModules; ++i) {
               auto& info = pollingInfo_[i];
               for (short j = 0; j < Config::NrConnectors; ++j) {
                  const short bit = 1 << j;
                  if (Config::ConnectorHasFeature(j,
                                                  ConnectorFeature::GainHV) &&
                      (overload[i] & bit) != (info.prevOverload & bit) &&
                      info.overloadChanged[j]) {
                     info.overloadChanged[j](overload[i] & bit);
                  }
                  if (Config::ConnectorHasFeature(
                          j, ConnectorFeature::CoolerCurrentLimit) &&
                      (currLmt[i] & bit) != (info.prevCurrLmt & bit) &&
                      info.currLmtChanged[j]) {
                     info.currLmtChanged[j](currLmt[i] & bit);
                  }
               }
               info.prevOverload = overload[i];
               info.prevCurrLmt = currLmt[i];
            }
         }
      });
   }

   void StopAndJoinPollingThread() {
      {
         std::lock_guard<std::mutex> lock(pollingMutex_);
         stopPollingRequested_ = true;
      }
      pollingCondVar_.notify_one();
      pollingThread_.join();
   }
};
