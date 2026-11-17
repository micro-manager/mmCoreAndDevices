///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI generic device shim providing the MM::Device boilerplate
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

#include "ASIBase.h"
#include "ASITiger.h"
#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceUtils.h"
#include <string>

// ---
// Class hierarchy (using CZStage as an example):
//
//   MM::Device
//     MM::Stage
//       CDeviceBase<MM::Stage, CZStage>
//         CStageBase<CZStage>   (MM framework CRTP)      ASIBase (plain class)
//                   \                                    /
//                    ASIDevice<CStageBase<CZStage>>   (this file)
//                      CZStage                        (concrete device)
//
// ASIDevice supplies the MM::Device boilerplate once for all devices: the
// constructor duties (name property, error messages, address parsing) and
// default implementations of Shutdown/GetName/Busy. Concrete devices only
// implement Initialize(), plus Busy() when the hardware has real busy logic.
//
// ASIBase holds the shared ASI state (hub pointer, firmware info, card
// address) and the serial-protocol helpers, which live in ASIBase.cpp so
// they compile once outside of any template.
//
// The hub uses the same template: ASIHub derives from
// ASIDevice<HubBase<ASIHub>> and leaves hub_ null.
// ---

template <typename T>
class ASIDevice : public T, public ASIBase {
public:
    explicit ASIDevice(const char* name) : deviceName_(name) {
        this->InitializeDefaultErrorMessages();
        InitializeASIErrorMessages();

        // Note: this name is different from the label that MM uses in e.g. SetProperty()

        // name property will be used to re-create the object by calling CreateDevice again with this parameter
        // if name isn't specified then skip this step
        if (!deviceName_.empty()) {
            this->CreateStringProperty(MM::g_Keyword_Name, name, true);
        }

        // sometimes constructor gets called without the full name like in the case of the hub
        //   so only set up these properties if we have the required information
        if (IsExtendedName(name)) {
            addressString_ = GetHexAddrFromExtName(name);
            if (addressString_ != g_EmptyCardAddressStr) {
                this->CreateStringProperty(g_TigerHexAddrPropertyName, addressString_.c_str(), true);
                addressChar_ = ConvertToTigerRawAddress(addressString_);
            }
        }
    }

    ~ASIDevice() override = default;

    // MM::Device API defaults; a concrete device overrides them only if its hardware requires it.
    int Shutdown() override {
        UnregisterFromHub(this);
        initialized_ = false;
        return DEVICE_OK;
    }

    void GetName(char* name) const override {
        CDeviceUtils::CopyLimitedString(name, deviceName_.empty() ? "Undefined" : deviceName_.c_str());
    }

    bool Busy() override {
        // The hub must keep this default: it is never busy and can always accept a new command or query.
        // Devices whose hardware can report busy override it.
        return false;
    }

protected:
    // The Initialize() of concrete peripheral classes must call this first.
    // Note that initialized_ is set by the concrete class.
    // Skip the firmware queries in special cases (when called with true flag, false is default) including
    //    FWheel: different serial terminator and command set => own Initialize() handles
    int PeripheralInitialize(bool skipFirmware = false) {
        if (const int status = AttachHub(this->GetParentHub()); status != DEVICE_OK) {
            return status;
        }

        // get the firmware version/date/build and expose them as read-only properties
        if (!skipFirmware) {
            if (const int status = QueryFirmwareInfo(); status != DEVICE_OK) {
                return status;
            }
            if (const int status = this->CreateFloatProperty(
                    g_FirmwareVersionPropertyName, firmwareVersion_, true); status != DEVICE_OK) {
                return status;
            }
            if (const int status = this->CreateStringProperty(
                    g_FirmwareDatePropertyName, firmwareDate_.c_str(), true); status != DEVICE_OK) {
                return status;
            }
            if (const int status = this->CreateStringProperty(
                    g_FirmwareBuildPropertyName, firmwareBuild_.c_str(), true); status != DEVICE_OK) {
                return status;
            }
        }

        RegisterWithHub(this);
        return DEVICE_OK;
    }

    void InitializeASIErrorMessages() {
        this->SetErrorText(ERR_UNRECOGNIZED_ANSWER, g_Msg_ERR_UNRECOGNIZED_ANSWER);
        this->SetErrorText(ERR_FILTER_WHEEL_NOT_READY, g_Msg_ERR_FILTER_WHEEL_NOT_READY);
        this->SetErrorText(ERR_FILTER_WHEEL_SPINNING, g_Msg_ERR_FILTER_WHEEL_SPINNING);
        this->SetErrorText(ERR_NOT_ENOUGH_AXES, g_Msg_ERR_NOT_ENOUGH_AXES);
        this->SetErrorText(ERR_TOO_LARGE_ADDRESSES, g_Msg_ERR_TOO_LARGE_ADDRESSES);
        this->SetErrorText(ERR_INFO_COMMAND_NOT_SUPPORTED, g_Msg_ERR_INFO_COMMAND_NOT_SUPPORTED);
        this->SetErrorText(ERR_TIGER_PAIR_NOT_PRESENT, g_Msg_ERR_TIGER_PAIR_NOT_PRESENT);
        this->SetErrorText(ERR_TIGER_DEV_NOT_SUPPORTED, g_Msg_ERR_TIGER_DEV_NOT_SUPPORTED);
        this->SetErrorText(ERR_CRISP_NOT_CALIBRATED, g_Msg_ERR_CRISP_NOT_CALIBRATED);
        this->SetErrorText(ERR_CRISP_NOT_LOCKED, g_Msg_ERR_CRISP_NOT_LOCKED);
        this->SetErrorText(ERR_UNKNOWN_COMMAND, g_Msg_ERR_UNKNOWN_COMMAND);
        this->SetErrorText(ERR_UNKNOWN_AXIS, g_Msg_ERR_UNKNOWN_AXIS);
        this->SetErrorText(ERR_MISSING_PARAM, g_Msg_ERR_MISSING_PARAM);
        this->SetErrorText(ERR_PARAM_OUT_OF_RANGE, g_Msg_ERR_PARAM_OUT_OF_RANGE);
        this->SetErrorText(ERR_OPERATION_FAILED, g_Msg_ERR_OPERATION_FAILED);
        this->SetErrorText(ERR_UNDEFINED_ERROR, g_Msg_ERR_UNDEFINED_ERROR);
        this->SetErrorText(ERR_INVALID_ADDRESS, g_Msg_ERR_INVALID_ADDRESS);
    }

    std::string deviceName_; // name passed to CreateDevice(), e.g. "ZStage:Z:32"; returned by GetName()
};
