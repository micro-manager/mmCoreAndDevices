///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIBase.h
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

#pragma once

#include "ASITiger.h"
#include "MMDevice.h"
#include <string>

class ASIHub;

////////////////////////////////////////////////////////////////
// State and helpers shared by every ASI Tiger device, including the hub.
// Devices inherit it through ASIDevice (see ASIDevice.h). It is not an
// MM::Device, so it must not declare Shutdown(), GetName() or Busy();
// those live in ASIDevice.
////////////////////////////////////////////////////////////////

class ASIBase {
public:
    ASIBase() = default;
    virtual ~ASIBase() = default;

    // Return true if the firmware version is at least the minimum version.
    bool FirmwareVersionAtLeast(const double minVersion) const {
        return firmwareVersion_ > (minVersion - 1e-6); // subtract a tiny epsilon to account for floating-point errors
    }

protected:
    // hub helpers
    int AttachHub(MM::Hub* genericHub);            // static_casts genericHub and stores hub_
    int QueryFirmwareInfo();                       // fills firmware fields below via the V/CD/BU commands
    void RegisterWithHub(const MM::Device* self);   // adds this device to the hub's lookup table
    void UnregisterFromHub(const MM::Device* self); // removes it again; safe to call if hub_ is null

    bool initialized_ = false;     // true once Initialize() has finished
    double firmwareVersion_ = 0.0; // firmware version
    std::string firmwareDate_{};   // firmware compile date
    std::string firmwareBuild_{};  // firmware build name

    ASIHub* hub_ = nullptr; // for serial communication; stays null for the hub device itself
    std::string addressString_ = g_EmptyCardAddressStr; // address within hub, in hex format, should be two characters (e.g. '31')
    std::string addressChar_ = g_EmptyCardAddressChar;  // address within hub, in single character (allowed to be extended ASCII so use string to store)
    bool refreshProps_ = false;    // true when property values should be read anew from controller each time
    bool refreshOverride_ = false; // true when device wants to manually force refreshes temporarily

    // helpers for "extended" device names containing address and axis letters
    static bool IsExtendedName(const char* name);
    static std::string GetAxisLetterFromExtName(const char* name, int position = 0); // returns single-character string
    static std::string GetHexAddrFromExtName(const char* name);
    static std::string ConvertToTigerRawAddress(const std::string& s);
    static int GetChannelFromExtName(const char* name);

private:
    static std::string ConvertTwoCharStringToHexChar(const std::string& s);
};
