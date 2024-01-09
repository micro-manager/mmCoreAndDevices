// Mock device adapter for testing of device change notifications
//
// Copyright (C) 2024 Board of Regents of the University of Wisconsin System
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
//
// Author: Mark A. Tsuchida

#include "DelayedNotifier.h"
#include "ProcessModel.h"

#include "DeviceBase.h"
#include "ModuleInterface.h"

#include <chrono>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

constexpr char DEVNAME_SYNC_PROPERTY[] = "NTSyncProperty";
constexpr char DEVNAME_ASYNC_PROPERTY[] = "NTAsyncProperty";
constexpr char PROPNAME_TEST_PROPERTY[] = "TestProperty";

template <typename ProcModel>
class NTestProp : public CGenericBase<NTestProp<ProcModel>>
{
   static constexpr bool isAsync_ =
         std::is_same<ProcModel, AsyncProcessModel>::value;

   std::string name_;
   ProcModel model_;
   DelayedNotifier delayer_;

   std::mutex notificationMut_;
   bool notificationsEnabled_ = false;

public:
   explicit NTestProp(std::string name) :
      name_(std::move(name)),
      model_([this](double pv) {
         this->LogMessage(("PV = " + std::to_string(pv)).c_str(), true);
         {
            std::lock_guard<std::mutex> lock(notificationMut_);
            if (!notificationsEnabled_) {
               return;
            }
         }
         delayer_.Schedule([this, pv = std::to_string(pv)]{
            this->OnPropertyChanged(PROPNAME_TEST_PROPERTY, pv.c_str());
         });
      })
   {}

   int Initialize() final {
      this->CreateStringProperty("NotificationsEnabled",
            notificationsEnabled_ ? "Yes" : "No", false,
            new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                        MM::ActionType eAct) {
               if (eAct == MM::BeforeGet) {
                  pProp->Set(notificationsEnabled_ ? "Yes" : "No");
               } else if (eAct == MM::AfterSet) {
                  std::string value;
                  pProp->Get(value);
                  notificationsEnabled_ = (value == "Yes");
               }
               return DEVICE_OK;
            }));
      this->AddAllowedValue("NotificationsEnabled", "No");
      this->AddAllowedValue("NotificationsEnabled", "Yes");

      this->CreateFloatProperty(PROPNAME_TEST_PROPERTY,
            model_.ProcessVariable(), false,
            new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                        MM::ActionType eAct) {
               if (eAct == MM::BeforeGet) {
                  pProp->Set(model_.ProcessVariable());
               } else if (eAct == MM::AfterSet) {
                  double value{};
                  pProp->Get(value);
                  this->LogMessage(
                        ("sp = " + std::to_string(value)).c_str(), true);
                  model_.Setpoint(value);
               }
               return DEVICE_OK;
            }));

      this->CreateFloatProperty("ExternallySet", model_.Setpoint(), false,
            new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                        MM::ActionType eAct) {
               if (eAct == MM::BeforeGet) {
                  // Keep last-set value
               } else if (eAct == MM::AfterSet) {
                  double value{};
                  pProp->Get(value);
                  this->LogMessage(("sp = " + std::to_string(value) +
                                    " (external)").c_str(), true);
                  model_.Setpoint(value);
               }
               return DEVICE_OK;
            }));

      if (isAsync_) {
         this->CreateFloatProperty("SlewTimePerUnit_s",
               model_.ReciprocalSlewRateSeconds(), false,
               new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                           MM::ActionType eAct) {
                  if (eAct == MM::BeforeGet) {
                     pProp->Set(model_.ReciprocalSlewRateSeconds());
                  } else if (eAct == MM::AfterSet) {
                     double seconds{};
                     pProp->Get(seconds);
                     model_.ReciprocalSlewRateSeconds(seconds);
                  }
                  return DEVICE_OK;
               }));
         this->SetPropertyLimits("SlewTimePerUnit_s", 0.0001, 10.0);

         this->CreateFloatProperty("UpdateInterval_s",
               model_.UpdateIntervalSeconds(), false,
               new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                           MM::ActionType eAct) {
                  if (eAct == MM::BeforeGet) {
                     pProp->Set(model_.UpdateIntervalSeconds());
                  } else if (eAct == MM::AfterSet) {
                     double seconds{};
                     pProp->Get(seconds);
                     model_.UpdateIntervalSeconds(seconds);
                  }
                  return DEVICE_OK;
               }));
         this->SetPropertyLimits("UpdateInterval_s", 0.0001, 10.0);

         using std::chrono::duration_cast;
         using std::chrono::microseconds;
         using FPSeconds = std::chrono::duration<double>;
         this->CreateFloatProperty("NotificationDelay_s",
               duration_cast<FPSeconds>(delayer_.Delay()).count(), false,
               new MM::ActionLambda([this](MM::PropertyBase* pProp,
                                           MM::ActionType eAct) {
                  if (eAct == MM::BeforeGet) {
                     pProp->Set(
                        duration_cast<FPSeconds>(delayer_.Delay()).count());
                  } else if (eAct == MM::AfterSet) {
                     double seconds{};
                     pProp->Get(seconds);
                     const auto delay = FPSeconds{seconds};
                     delayer_.Delay(duration_cast<microseconds>(delay));
                  }
                  return DEVICE_OK;
               }));
         this->SetPropertyLimits("NotificationDelay_s", 0.0, 1.0);
      }

      return DEVICE_OK;
   }

   int Shutdown() final {
      model_.Halt();
      delayer_.CancelAll();
      return DEVICE_OK;
   }

   bool Busy() final {
      return model_.IsSlewing();
   }

   void GetName(char *name) const final {
      CDeviceUtils::CopyLimitedString(name, name_.c_str());
   }
};

MODULE_API void InitializeModuleData()
{
   RegisterDevice(DEVNAME_SYNC_PROPERTY, MM::GenericDevice,
                  "Test synchronous property notifications");
   RegisterDevice(DEVNAME_ASYNC_PROPERTY, MM::GenericDevice,
                  "Test asynchronous property busy state and notifications");
}

MODULE_API MM::Device *CreateDevice(const char *deviceName)
{
   const std::string name(deviceName);
   if (name == DEVNAME_SYNC_PROPERTY)
      return new NTestProp<SyncProcessModel>(name);
   if (name == DEVNAME_ASYNC_PROPERTY)
      return new NTestProp<AsyncProcessModel>(name);
   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device *pDevice)
{
   delete pDevice;
}