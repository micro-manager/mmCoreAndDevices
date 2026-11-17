///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIBase.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   State and helpers shared by all ASI Tiger devices
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

#include "ASIBase.h"
#include "ASIHub.h"
#include "DeviceUtils.h"
#include <cstdlib>
#include <sstream>
#include <vector>

int ASIBase::AttachHub(MM::Hub* genericHub) {
    // GetParentHub() only returns a hub from this library, and every ASITiger hub derives from ASIHub
    hub_ = static_cast<ASIHub*>(genericHub);
    return hub_ ? DEVICE_OK : DEVICE_COMM_HUB_MISSING;
}

int ASIBase::QueryFirmwareInfo() {
    // firmware version
    if (const int status = hub_->QueryCommandVerify(addressChar_ + "V", ":A v"); status != DEVICE_OK) {
        return status;
    }
    if (const int status = hub_->ParseAnswerAfterPosition(4, firmwareVersion_); status != DEVICE_OK) {
        return status;
    }

    // firmware compile date; currently for user information only, stored as a string
    // so would need to convert it to proper type to use in comparisons, etc.
    if (const int status = hub_->QueryCommand(addressChar_ + "CD"); status != DEVICE_OK) {
        return status;
    }
    firmwareDate_ = hub_->LastSerialAnswer();

    // firmware build name
    if (const int status = hub_->QueryCommand(addressChar_ + "BU"); status != DEVICE_OK) {
        return status;
    }
    firmwareBuild_ = hub_->LastSerialAnswer();

    return DEVICE_OK;
}

void ASIBase::RegisterWithHub(const MM::Device* self) {
    char deviceLabel[MM::MaxStrLength];
    self->GetLabel(deviceLabel);
    hub_->RegisterPeripheral(deviceLabel, addressChar_);
}

void ASIBase::UnregisterFromHub(const MM::Device* self) {
    // NB: can be called even if the device was never fully initialized
    if (hub_) {
        char deviceLabel[MM::MaxStrLength];
        self->GetLabel(deviceLabel);
        hub_->UnRegisterPeripheral(deviceLabel);
    }
}

bool ASIBase::IsExtendedName(const char* name) {
    std::vector<std::string> vName;
    CDeviceUtils::Tokenize(name, vName, ":");
    return vName.size() > 2;
}

std::string ASIBase::GetAxisLetterFromExtName(const char* name, int position) {
    std::vector<std::string> vName;
    CDeviceUtils::Tokenize(name, vName, ":");
    if (vName.size() > 1) {
        return vName[1].substr(position, 1);
    }
    return g_EmptyAxisLetterStr;
}

std::string ASIBase::GetHexAddrFromExtName(const char* name) {
    std::vector<std::string> vName;
    CDeviceUtils::Tokenize(name, vName, ":");
    if (vName.size() > 2) {
        return vName[2];
    }
    return g_EmptyCardAddressStr;
}

int ASIBase::GetChannelFromExtName(const char* name) {
    std::vector<std::string> vName;
    CDeviceUtils::Tokenize(name, vName, ":");
    if (vName.size() > 3) {
        return atoi(vName[3].c_str());
    }
    return 0; // if no channel is provided (LED on 2-axis card) return 0
}

// does the dirty work of converting a two-character hex (e.g. F5) into the single character
// only works for valid TG-1000 addresses
// see ConvertToTigerRawAddress comments for more details
std::string ASIBase::ConvertTwoCharStringToHexChar(const std::string& s) {
    if (s.size() != 2) {
        return g_EmptyCardAddressCode;
    }

    unsigned int code;
    std::stringstream ss;
    ss << std::hex << s;
    ss >> code;
    if ((code >= 0x31 && code <= 0x39) || (code >= 0x81 && code <= 0xF5)) {
        ss.str("");
        ss.clear(); // clears hex flag
        ss << (char) code;
        std::string s2;
        ss >> s2;
        return s2;
    }
    return g_EmptyCardAddressCode;
}

std::string ASIBase::ConvertToTigerRawAddress(const std::string& s) {
    // Tiger addresses are 0x31 to 0x39 and then 0x81 to 0xF5 (i.e. '1' to '9' and then extended ASCII)
    // these addresses are prepended (in extended ASCII char) to serial commands that are addressed
    // MM doesn't handle names with extended ASCII, and we need to include address in MM's device name
    // so in MM device names we represent the address in 2-digit hex string (31..39, 81..F5)
    // use this function to get the serial char to send from a MM 2-digit hex string
    // returns g_EmptyCardAddressCode in case of error (set to space for minimum impact on Tiger operation)

    // make sure the number of characters is even and strictly positive
    if ((s.size() == 0) || (s.size() % 2)) {
        return g_EmptyCardAddressCode;
    }

    std::string s2 = "";
    for (std::string::size_type iii = 0; iii < s.size()/2; ++iii) {
        // operate on chunks of two characters
        s2.append(ConvertTwoCharStringToHexChar(s.substr(iii*2, 2)));
    }
    return s2;
}
