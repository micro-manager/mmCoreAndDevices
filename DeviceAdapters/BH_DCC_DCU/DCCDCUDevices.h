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

#include "DCCDCUInterface.h"

#include "DeviceBase.h"

#include <cassert>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>

// This can be moved to DeviceBase.h to make available generally.
class ActionLambda final : public MM::ActionFunctor {
   std::function<int(MM::PropertyBase*, MM::ActionType)> f_;

 public:
   template <typename F> explicit ActionLambda(F f) : f_(f) {}

   auto Execute(MM::PropertyBase* pProp, MM::ActionType eAct) -> int {
      return f_(pProp, eAct);
   }
};

template <DCCOrDCU Model> inline auto ModelName() -> std::string {
   if (Model == DCCOrDCU::DCC) {
      return "DCC";
   } else if (Model == DCCOrDCU::DCU) {
      return "DCU";
   } else {
      assert(false);
   }
}

template <DCCOrDCU Model> inline auto ModelDescription() -> std::string {
   if (Model == DCCOrDCU::DCC) {
      return "Becker & Hickl DCC-100 detector control cards";
   } else if (Model == DCCOrDCU::DCU) {
      return "Becker & Hickl DCU USB detector control units";
   } else {
      assert(false);
   }
}

enum class Errors {
   CANNOT_CREATE_TWO_HUBS = 20001,
   PRE_INIT_FAILURE,
   INIT_FAILED,
   CANNOT_GET_INIT_STATUS,
   MODULE_IN_USE,
   MODULE_INIT_MISC_FAILURE,
   CANNOT_GET_MODULE_INFO,
   CANNOT_GET_PARAMETER,
   CANNOT_SET_PARAMETER,
   CANNOT_ENABLE_OUTPUTS,
   CANNOT_CLEAR_OVERLOADS,
   CANNOT_GET_OVERLOAD_STATE,
   CANNOT_GET_COOLER_CURRENT_LIMIT_STATE,
};

template <DCCOrDCU Model, typename SetFunc>
inline void RegisterErrorMessages(SetFunc f) {
   auto f2 = [&](Errors code, std::string const& text) {
      f(static_cast<int>(code), text.c_str());
   };
   const std::string model = ModelName<Model>();
   f2(Errors::CANNOT_CREATE_TWO_HUBS,
      "Cannot create more than one " + model + " hub at the same time");
   f2(Errors::PRE_INIT_FAILURE,
      "Could not initialize " + model +
          " because a temporary .ini file could not be created");
   f2(Errors::INIT_FAILED,
      model + " init failed (see CoreLog for internal details)");
   f2(Errors::CANNOT_GET_INIT_STATUS,
      "Failed to get " + model + " init status");
   f2(Errors::MODULE_IN_USE, model +
                                 " module is in use by another application "
                                 "(or needs to be reset using the BH " +
                                 model + " app)");
   f2(Errors::MODULE_INIT_MISC_FAILURE,
      model + " module init failed (see CoreLog for internal details)");
   f2(Errors::CANNOT_GET_MODULE_INFO,
      "Failed to get " + model + " module info");
   f2(Errors::CANNOT_GET_PARAMETER, "Failed to get " + model + " parameter");
   f2(Errors::CANNOT_SET_PARAMETER, "Failed to set " + model + " parameter");
   f2(Errors::CANNOT_ENABLE_OUTPUTS,
      "Failed to enable or disable " + model + " outputs");
   f2(Errors::CANNOT_CLEAR_OVERLOADS, "Failed to clear overloads");
   f2(Errors::CANNOT_GET_OVERLOAD_STATE,
      "Failed to get detector overload status");
   f2(Errors::CANNOT_GET_COOLER_CURRENT_LIMIT_STATE,
      "Failed to get cooler current limit status");
}

template <DCCOrDCU Model> inline auto HubDeviceName() -> std::string {
   return ModelName<Model>() + "Hub";
}

template <DCCOrDCU Model> inline auto ModuleDeviceNamePrefix() -> std::string {
   return ModelName<Model>() + "Module";
}

template <DCCOrDCU Model>
inline auto ModuleDeviceName(short moduleNo) -> std::string {
   return ModuleDeviceNamePrefix<Model>() + std::to_string(moduleNo + 1);
}

// Return module no if name matches (prefix + number); else return -1.
template <DCCOrDCU Model>
inline auto DeviceNameToModuleNo(std::string const& name) -> short {
   const std::string prefix = ModuleDeviceNamePrefix<Model>();
   if (name.rfind(prefix, 0) != 0) {
      return -1;
   }
   const auto index = name.substr(prefix.size());
   std::size_t len = 0;
   std::size_t i = 0;
   try {
      i = std::stoul(index, &len);
   } catch (std::exception const&) {
      return -1;
   }
   if (len < index.size()) {
      return -1;
   }
   if (i == 0 || i > DCCDCUConfig<Model>::MaxNrModules) {
      return -1;
   }
   return static_cast<short>(i - 1);
}

template <DCCOrDCU Model>
class DCCDCUHubDevice : public HubBase<DCCDCUHubDevice<Model>> {
   using Config = DCCDCUConfig<Model>;
   std::shared_ptr<DCCDCUInterface<Model>> interface_;
   std::string deviceName_;

   // Forbid simultaneous creation of more than one instance (per model).
   static int instanceCount_;

 public:
   DCCDCUHubDevice(std::string name) : deviceName_(std::move(name)) {
      RegisterErrorMessages<Model>([this](int code, const char* text) {
         this->SetErrorText(code, text);
      });

      this->CreateStringProperty("SimulateDevice", "No", false, nullptr, true);
      this->AddAllowedValue("SimulateDevice", "No");
      this->AddAllowedValue("SimulateDevice", "Yes");
      for (short i = 0; i < Config::MaxNrModules; ++i) {
         const auto prop = "UseModule" + std::to_string(i + 1);
         this->CreateStringProperty(prop.c_str(), i == 0 ? "Yes" : "No", false,
                                    nullptr, true);
         this->AddAllowedValue(prop.c_str(), "No");
         this->AddAllowedValue(prop.c_str(), "Yes");
      }
   }

   auto Initialize() -> int final {
      if (instanceCount_ >= 1) {
         return static_cast<int>(Errors::CANNOT_CREATE_TWO_HUBS);
      }
      ++instanceCount_;

      const bool simulate = RequestedSimulation();
      const auto moduleSet = RequestedModuleSet();
      interface_ =
          std::make_shared<DCCDCUInterface<Model>>(moduleSet, simulate);
      if (interface_->PreInitError()) {
         return static_cast<int>(Errors::PRE_INIT_FAILURE);
      }

      const auto initErr = interface_->InitError();
      if (initErr) {
         this->LogMessage(ModelName<Model>() + " init failed with error: " +
                          DCCDCUGetErrorString(initErr));
      }
      // It is not clear whether we will get an error when some, but not all,
      // of the modules failed to initialize. So generally we just log and
      // continue and handle the init status in each module. But let's give
      // up when it is clearly an error not specific to individual modules.
      switch (initErr) {
      case DCC_OPEN_FILE:
      case DCC_FILE_NVALID:
      case DCC_MEM_ALLOC:
      case DCC_READ_STR:
         return static_cast<int>(Errors::INIT_FAILED);
      }

      CreateProperties();
      return DEVICE_OK;
   }

   auto Shutdown() -> int final {
      interface_.reset();
      --instanceCount_;
      return DEVICE_OK;
   }

   void GetName(char* name) const final {
      CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
   }
   auto Busy() -> bool final { return false; }

   auto DetectInstalledDevices() -> int final {
      const auto moduleSet = RequestedModuleSet();
      for (short i = 0; i < Config::MaxNrModules; ++i) {
         if (moduleSet[i]) {
            this->AddInstalledDevice(
                CreateDevice(ModuleDeviceName<Model>(i).c_str()));
         }
      }
      return DEVICE_OK;
   }

   auto GetDCCDCUInterface() -> std::shared_ptr<DCCDCUInterface<Model>> {
      return interface_;
   }

 private:
   auto RequestedSimulation() const -> bool {
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty("SimulateDevice", buf.data());
      const std::string yesno = buf.data();
      return yesno == "Yes";
   }

   auto RequestedModuleSet() const -> std::bitset<Config::MaxNrModules> {
      std::bitset<Config::MaxNrModules> ret;
      for (short i = 0; i < Config::MaxNrModules; ++i) {
         const auto prop = "UseModule" + std::to_string(i + 1);
         std::array<char, MM::MaxStrLength + 1> buf;
         this->GetProperty(prop.c_str(), buf.data());
         const std::string yesno = buf.data();
         if (yesno == "Yes") {
            ret.set(i);
         }
      }
      return ret;
   }

   void CreateProperties() {
      this->CreateStringProperty(
          "Simulated", interface_->IsSimulating() ? "Yes" : "No", true);
   }
};

template <DCCOrDCU Model> int DCCDCUHubDevice<Model>::instanceCount_ = 0;

template <DCCOrDCU Model>
class DCCDCUModuleDevice : public CGenericBase<DCCDCUModuleDevice<Model>> {
   using Config = DCCDCUConfig<Model>;
   std::shared_ptr<DCCDCUModule<Model>> module_;
   short moduleNo_;
   std::string deviceName_;

 public:
   DCCDCUModuleDevice(std::string name, short moduleNo)
       : moduleNo_(moduleNo), deviceName_(std::move(name)) {
      RegisterErrorMessages<Model>([this](int code, const char* text) {
         this->SetErrorText(code, text);
      });
   }

   auto Initialize() -> int final {
      auto* hub = static_cast<DCCDCUHubDevice<Model>*>(this->GetParentHub());
      auto iface = hub->GetDCCDCUInterface();

      module_ = iface->GetModule(moduleNo_);

      const bool isActive = module_->IsActive();
      this->LogMessage(ModelName<Model>() + " module " +
                       (isActive ? "is" : "is NOT") + " active");

      short err{};
      const auto initStatus = module_->InitStatus(err);
      if (err) {
         return static_cast<int>(Errors::CANNOT_GET_INIT_STATUS);
      }
      this->LogMessage(ModelName<Model>() + " module init status is " +
                       std::to_string(initStatus));
      switch (initStatus) {
      case INIT_DCC_OK:
         break;
      case INIT_DCC_MOD_IN_USE:
         return static_cast<int>(Errors::MODULE_IN_USE);
      default:
         return static_cast<int>(Errors::MODULE_INIT_MISC_FAILURE);
      }

      const auto modInfo = module_->ModInfo(err);
      if (err) {
         return static_cast<int>(Errors::CANNOT_GET_MODULE_INFO);
      }

      CreateModInfoProperties(modInfo);
      CreateConnectorProperties();
      if (not Config::HasIndividualOutputAndClearOverload) {
         CreateEnableAllOutputsProperty();
         CreateClearAllOverloadsProperty();
      }
      return DEVICE_OK;
   }

   auto Shutdown() -> int final {
      module_->Close();
      return DEVICE_OK;
   }

   void GetName(char* name) const final {
      CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
   }

   auto Busy() -> bool final { return false; }

 private:
   void CreateModInfoProperties(typename Config::ModInfoType modInfo) {
      this->CreateIntegerProperty("ModuleNumber", moduleNo_ + 1, true);
      std::string model;
      switch (modInfo.ModuleType()) {
      case M_DCC100:
         model = "DCC-100";
         break;
      case M_DCCUSB:
         model = "DCC-USB (DCU)";
         break;
      default:
         model = std::to_string(modInfo.ModuleType());
         break;
      }
      this->CreateStringProperty("ModuleType", model.c_str(), true);
      if (Model == DCCOrDCU::DCC) {
         this->CreateIntegerProperty("BusNumber", modInfo.BusNumber(), true);
         this->CreateIntegerProperty("SlotNumber", modInfo.SlotNumber(), true);
         this->CreateIntegerProperty("BaseAddress", modInfo.BaseAdr(), true);
      }
      if (Model == DCCOrDCU::DCU) {
         this->CreateIntegerProperty("ComPortNumber", modInfo.ComPortNo(),
                                     true);
      }
      this->CreateStringProperty("SerialNumber", modInfo.SerialNo().c_str(),
                                 true);
   }

   void CreateConnectorProperties() {
      for (short i = 0; i < Config::NrConnectors; ++i) {
         CreateConnectorProperties(i);
      }
   }

   void CreateConnectorProperties(short connNo) {
      const std::string propPrefix = "C" + std::to_string(connNo + 1) + "_";
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::Plus5V)) {
         CreateConnectorOnOffProperty(connNo, ConnectorFeature::Plus5V,
                                      propPrefix + "Plus5V");
      }
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::Minus5V)) {
         CreateConnectorOnOffProperty(connNo, ConnectorFeature::Minus5V,
                                      propPrefix + "Minus5V");
      }
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::Plus12V)) {
         CreateConnectorOnOffProperty(connNo, ConnectorFeature::Plus12V,
                                      propPrefix + "Plus12V");
      }
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::GainHV)) {
         // The GainHV limit is set in the EEPROM, but in MM we treat it as
         // a permanent limit. User can configure using the BH app.
         short err{};
         float lim = module_->GetGainHVLimit(connNo, err);
         if (err) // Unlikely
            lim = 100.0f;
         CreateConnectorFloatProperty(connNo, ConnectorFeature::GainHV,
                                      propPrefix + "GainHV", 0.0f, lim);
         CreateConnectorOverloadedProperty(connNo, propPrefix + "Overloaded");
         if (Config::HasIndividualOutputAndClearOverload) {
            CreateConnectorClearOverloadProperty(connNo,
                                                 propPrefix + "ClearOverload");
         }
      }
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::DigitalOut)) {
         CreateConnectorByteProperty(connNo, ConnectorFeature::DigitalOut,
                                     propPrefix + "DigitalOut");
      }
      if (Config::ConnectorHasFeature(connNo, ConnectorFeature::Cooling)) {
         CreateConnectorOnOffProperty(connNo, ConnectorFeature::Cooling,
                                      propPrefix + "Cooling");
      }
      if (Config::ConnectorHasFeature(connNo,
                                      ConnectorFeature::CoolerVoltage)) {
         CreateConnectorFloatProperty(connNo, ConnectorFeature::CoolerVoltage,
                                      propPrefix + "CoolerVoltage", 0.0f,
                                      5.0f);
      }
      if (Config::ConnectorHasFeature(connNo,
                                      ConnectorFeature::CoolerCurrentLimit)) {
         CreateConnectorFloatProperty(
             connNo, ConnectorFeature::CoolerCurrentLimit,
             propPrefix + "CoolerCurrentLimit", 0.0f, 2.0f);
         CreateConnectorCurrentLimitReachedProperty(
             connNo, propPrefix + "CoolerCurrentLimitReached");
      }
      if (Config::HasIndividualOutputAndClearOverload) {
         CreateConnectorEnableOutputProperty(connNo,
                                             propPrefix + "EnableOutputs");
      }
   }

   void CreateConnectorOnOffProperty(short connNo, ConnectorFeature feature,
                                     const std::string& name) {
      this->CreateStringProperty(
          name.c_str(), "Off", false,
          new ActionLambda([this, connNo, feature, name](
                               MM::PropertyBase* pProp, MM::ActionType eAct) {
             if (eAct == MM::BeforeGet) {
                short err{};
                const bool flag =
                    module_->GetConnectorParameterBool(connNo, feature, err);
                if (err) {
                   this->LogMessage("Cannot get parameter for property " +
                                    name + ": " + DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_GET_PARAMETER);
                }
                pProp->Set(flag ? "On" : "Off");
             } else if (eAct == MM::AfterSet) {
                std::string propVal;
                pProp->Get(propVal);
                const bool value = propVal == "On";
                short err{};
                module_->SetConnectorParameterBool(connNo, feature, value,
                                                   err);
                if (err) {
                   this->LogMessage("Cannot set parameter for property " +
                                    name + " to value " +
                                    std::to_string(value) + ": " +
                                    DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_SET_PARAMETER);
                }
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue(name.c_str(), "Off");
      this->AddAllowedValue(name.c_str(), "On");
      // Load current value:
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty(name.c_str(), buf.data());
   }

   void CreateConnectorFloatProperty(short connNo, ConnectorFeature feature,
                                     const std::string& name, float minValue,
                                     float maxValue) {
      this->CreateFloatProperty(
          name.c_str(), minValue, false,
          new ActionLambda([this, connNo, feature, name](
                               MM::PropertyBase* pProp, MM::ActionType eAct) {
             if (eAct == MM::BeforeGet) {
                short err{};
                const float value =
                    module_->GetConnectorParameterFloat(connNo, feature, err);
                if (err) {
                   this->LogMessage("Cannot get parameter for property " +
                                    name + ": " + DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_GET_PARAMETER);
                }
                pProp->Set(value);
             } else if (eAct == MM::AfterSet) {
                double propVal{};
                pProp->Get(propVal);
                const auto value = static_cast<float>(propVal);
                short err{};
                module_->SetConnectorParameterFloat(connNo, feature, value,
                                                    err);
                if (err) {
                   this->LogMessage("Cannot set parameter for property " +
                                    name + " to value " +
                                    std::to_string(value) + ": " +
                                    DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_SET_PARAMETER);
                }
             }
             return DEVICE_OK;
          }));
      this->SetPropertyLimits(name.c_str(), minValue, maxValue);
      // Load current value:
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty(name.c_str(), buf.data());
   }

   void CreateConnectorByteProperty(short connNo, ConnectorFeature feature,
                                    const std::string& name) {
      this->CreateIntegerProperty(
          name.c_str(), 0, false,
          new ActionLambda([this, connNo, feature, name](
                               MM::PropertyBase* pProp, MM::ActionType eAct) {
             if (eAct == MM::BeforeGet) {
                short err{};
                const unsigned value =
                    module_->GetConnectorParameterUInt(connNo, feature, err);
                if (err) {
                   this->LogMessage("Cannot get parameter for property " +
                                    name + ": " + DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_GET_PARAMETER);
                }
                pProp->Set(static_cast<long>(value));
             } else if (eAct == MM::AfterSet) {
                long propVal{};
                pProp->Get(propVal);
                const auto value = static_cast<unsigned>(propVal);
                short err{};
                module_->SetConnectorParameterUInt(connNo, feature, value,
                                                   err);
                if (err) {
                   this->LogMessage("Cannot set parameter for property " +
                                    name + " to value " +
                                    std::to_string(value) + ": " +
                                    DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_SET_PARAMETER);
                }
             }
             return DEVICE_OK;
          }));
      for (long v = 0; v < 256; ++v) {
         this->AddAllowedValue(name.c_str(), std::to_string(v).c_str());
      }
      // Load current value:
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty(name.c_str(), buf.data());
   }

   void CreateConnectorEnableOutputProperty(short connNo,
                                            const std::string& name) {
      this->CreateStringProperty(
          name.c_str(), "Off", false,
          new ActionLambda([this, connNo](MM::PropertyBase* pProp,
                                          MM::ActionType eAct) {
             // There is no readout for enable_outputs, so we rely on the
             // last-set value.
             if (eAct == MM::AfterSet) {
                std::string propVal;
                pProp->Get(propVal);
                const bool value = propVal == "On";
                short err{};
                module_->EnableConnectorOutputs(connNo, value, err);
                if (err) {
                   pProp->Set("Unknown");
                   using namespace std::string_literals;
                   this->LogMessage("Cannot "s +
                                    (value ? "enable" : "disable") +
                                    " outputs: " + DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_ENABLE_OUTPUTS);
                }
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue(name.c_str(), "Off");
      this->AddAllowedValue(name.c_str(), "On");
      // Force off initially to ensure sync with property value:
      this->SetProperty(name.c_str(), "Off");
   }

   void CreateEnableAllOutputsProperty() {
      this->CreateStringProperty(
          "EnableOutputs", "Off", false,
          new ActionLambda([this](MM::PropertyBase* pProp,
                                  MM::ActionType eAct) {
             // There is no readout for enable_outputs, so we rely on the
             // last-set value.
             if (eAct == MM::AfterSet) {
                std::string propVal;
                pProp->Get(propVal);
                const bool value = propVal == "On";
                short err{};
                module_->EnableAllOutputs(value, err);
                if (err) {
                   pProp->Set("Unknown");
                   using namespace std::string_literals;
                   this->LogMessage("Cannot "s +
                                    (value ? "enable" : "disable") +
                                    " outputs: " + DCCDCUGetErrorString(err));

                   return static_cast<int>(Errors::CANNOT_ENABLE_OUTPUTS);
                }
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue("EnableOutputs", "Off");
      this->AddAllowedValue("EnableOutputs", "On");
      // Force off initially to ensure sync with property value:
      this->SetProperty("EnableOutputs", "Off");
   }

   void CreateConnectorClearOverloadProperty(short connNo,
                                             const std::string& name) {
      this->CreateStringProperty(
          name.c_str(), "", false,
          new ActionLambda([this, connNo](MM::PropertyBase* pProp,
                                          MM::ActionType eAct) {
             if (eAct == MM::AfterSet) {
                std::string propVal;
                pProp->Get(propVal);
                if (propVal == "Clear") {
                   short err{};
                   module_->ClearConnectorOverload(connNo, err);
                   if (err) {
                      this->LogMessage("Cannot clear overloads: " +
                                       DCCDCUGetErrorString(err));
                      return static_cast<int>(Errors::CANNOT_CLEAR_OVERLOADS);
                   }
                   pProp->Set("");
                }
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue(name.c_str(), "");
      this->AddAllowedValue(name.c_str(), "Clear");
   }

   void CreateClearAllOverloadsProperty() {
      this->CreateStringProperty(
          "ClearOverloads", "", false,
          new ActionLambda([this](MM::PropertyBase* pProp,
                                  MM::ActionType eAct) {
             if (eAct == MM::AfterSet) {
                std::string propVal;
                pProp->Get(propVal);
                if (propVal == "Clear") {
                   short err{};
                   module_->ClearAllOverloads(err);
                   if (err) {
                      this->LogMessage("Cannot clear overloads: " +
                                       DCCDCUGetErrorString(err));
                      return static_cast<int>(Errors::CANNOT_CLEAR_OVERLOADS);
                   }
                   pProp->Set("");
                }
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue("ClearOverloads", "");
      this->AddAllowedValue("ClearOverloads", "Clear");
   }

   void CreateConnectorOverloadedProperty(short connNo,
                                          const std::string& name) {
      this->CreateStringProperty(
          name.c_str(), "No", true,
          new ActionLambda([this, connNo](MM::PropertyBase* pProp,
                                          MM::ActionType eAct) {
             if (eAct == MM::BeforeGet) {
                short err{};
                const bool flag = module_->IsOverloaded(connNo, err);
                if (err) {
                   this->LogMessage("Cannot get overload state: " +
                                    DCCDCUGetErrorString(err));
                   return static_cast<int>(Errors::CANNOT_GET_OVERLOAD_STATE);
                }
                pProp->Set(flag ? "Yes" : "No");
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue(name.c_str(), "No");
      this->AddAllowedValue(name.c_str(), "Yes");

      // Subscribe to changes:
      module_->SetOverloadChangeHandler(connNo, [this, name](bool overloaded) {
         this->OnPropertyChanged(name.c_str(), (overloaded ? "Yes" : "No"));
      });

      // Load current value:
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty(name.c_str(), buf.data());
   }

   void CreateConnectorCurrentLimitReachedProperty(short connNo,
                                                   const std::string& name) {
      this->CreateStringProperty(
          name.c_str(), "No", true,
          new ActionLambda([this, connNo](MM::PropertyBase* pProp,
                                          MM::ActionType eAct) {
             if (eAct == MM::BeforeGet) {
                short err{};
                const bool flag =
                    module_->IsCoolerCurrentLimitReached(connNo, err);
                if (err) {
                   this->LogMessage("Cannot get cooler current limit state: " +
                                    DCCDCUGetErrorString(err));
                   return static_cast<int>(
                       Errors::CANNOT_GET_COOLER_CURRENT_LIMIT_STATE);
                }
                pProp->Set(flag ? "Yes" : "No");
             }
             return DEVICE_OK;
          }));
      this->AddAllowedValue(name.c_str(), "No");
      this->AddAllowedValue(name.c_str(), "Yes");

      // Subscribe to changes:
      module_->SetCurrLmtChangeHandler(connNo, [this,
                                                name](bool limitReached) {
         this->OnPropertyChanged(name.c_str(), (limitReached ? "Yes" : "No"));
      });

      // Load current value:
      std::array<char, MM::MaxStrLength + 1> buf;
      this->GetProperty(name.c_str(), buf.data());
   }
};
