///////////////////////////////////////////////////////////////////////////////
// FILE:          WPTR.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   RND's WTR controller adapter
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//                Robots and Design Co, Ltd.
//                University of California, San Francisco
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER(S) OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Vikram Kopuri, based on Code by Nenad Amodaj Nico Stuurman and Jizhen Zhao
//

#include "wptr.h"

#include <string>

static const char* gWPTRobotName = "WPTRobot";

// Exported MMDevice API
MODULE_API void InitializeModuleData() {
    RegisterDevice(gWPTRobotName, MM::GenericDevice, "WPTRobot");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName) {
    if (deviceName == nullptr) {
        return nullptr;
    }

    if (std::string(deviceName) == gWPTRobotName) {
        return new WPTRobot();
    }

    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
    delete pDevice;
}

// Error codes
static constexpr int ERR_UNKNOWN_COMMAND = 10002;
static constexpr int ERR_PORT_CHANGE_FORBIDDEN = 10004;
static constexpr int ERR_UNRECOGNIZED_ANSWER = 10009;

// Clear contents of serial port
static int ClearPort(const MM::Device& device, MM::Core& core, const std::string& port) {
    constexpr size_t bufferSize = 255;
    unsigned char clear[bufferSize];
    unsigned long read = bufferSize;
    while (read == bufferSize) {
        if (const int status = core.ReadFromSerial(&device, port.c_str(), clear, bufferSize, read);
            status != DEVICE_OK) {
            return status;
        }
    }
    return DEVICE_OK;
}

// Returns true if the string starts with the prefix.
static bool StartsWith(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.size(), prefix) == 0;
}

WPTRobot::WPTRobot() {
    InitializeDefaultErrorMessages();

    // register custom errors
    SetErrorText(ERR_UNKNOWN_COMMAND, "Unknown serial command, not ORG, GET, PUT, AES, or DRT");
    SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Serial port cannot be changed after initialization");
    SetErrorText(ERR_UNRECOGNIZED_ANSWER, "Serial command replied with an unrecognized answer");

    // create pre-init properties
    CreateProperty(MM::g_Keyword_Name, gWPTRobotName, MM::String, true);
    CreateProperty(MM::g_Keyword_Description, "Wellplate Transfer Robot", MM::String, true);
    CreatePortProperty();
}

WPTRobot::~WPTRobot() {
    Shutdown();
}

void WPTRobot::GetName(char* name) const {
    CDeviceUtils::CopyLimitedString(name, gWPTRobotName);
}

int WPTRobot::Initialize() {
    // empty the serial rx buffer before sending command
    ClearPort(*this, *GetCoreCallback(), port_.c_str());

    // stage and slot are passed as properties
    CreateStageProperty();
    CreateSlotProperty();

    // setting this property to different keywords leads to a command
    CreateCommandProperty();

    // update all properties
    if (const int status = UpdateStatus(); status != DEVICE_OK) {
        return status;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int WPTRobot::Shutdown() {
    initialized_ = false;
    return DEVICE_OK;
}

bool WPTRobot::Busy() {
    return false; // reply is only given after move is completed, so Busy() not necessary
}

// Returns the device status after sending a serial command.
int WPTRobot::SendCommand(const std::string& command) {
    // send the command
    if (const int status = SendSerialCommand(port_.c_str(), command.c_str(), "\r\n");
        status != DEVICE_OK) {
        return status;
    }
    // read the answer
    std::string answer;
    if (const int status = GetSerialAnswer(port_.c_str(), "\r\n", answer);
        status != DEVICE_OK) {
        return status;
    }
    // Note: protocol specific, reply length is always 3, matches command length
    if (answer.length() != 3 || answer.compare(0, 3, command) != 0) {
        return ERR_UNRECOGNIZED_ANSWER;
    }
    return DEVICE_OK;
}

void WPTRobot::CreatePortProperty() {
    CreateStringProperty(MM::g_Keyword_Port, "Undefined", false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(port_.c_str());
            } else if (eAct == MM::AfterSet) {
                if (initialized_) {
                    // revert change
                    pProp->Set(port_.c_str());
                    return ERR_PORT_CHANGE_FORBIDDEN;
                }
                pProp->Get(port_);
            }
            return DEVICE_OK;
        }),
        true // pre-init property
    );
}

// Just reading and writing to the stage variable
void WPTRobot::CreateStageProperty() {
    CreateIntegerProperty("Stage", 1, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(stage_);
            } else if (eAct == MM::AfterSet) {
                pProp->Get(stage_);
            }
            return DEVICE_OK;
        })
    );
}

// Just reading and writing to the slot variable
void WPTRobot::CreateSlotProperty() {
    CreateIntegerProperty("Slot", 1, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(slot_);
            } else if (eAct == MM::AfterSet) {
                pProp->Get(slot_);
            }
            return DEVICE_OK;
        })
    );
}

void WPTRobot::CreateCommandProperty() {
    CreateStringProperty("Command", "Undefined", false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(command_.c_str());
            } else if (eAct == MM::AfterSet) {
                pProp->Get(command_);

                // is the command valid?
                if (!StartsWith(command_, "ORG")
                    && !StartsWith(command_, "GET")
                    && !StartsWith(command_, "PUT")
                    && !StartsWith(command_, "AES")    // command for emergency stop
                    && !StartsWith(command_, "DRT")) { // command to enable after AES
                    return ERR_UNKNOWN_COMMAND;
                }

                // prefix is a valid command after check above
                const std::string prefix = command_.substr(0, 3);

                // append stage and slot for GET and PUT commands
                const std::string command = (prefix == "GET" || prefix == "PUT")
                    ? prefix + " " + std::to_string(stage_) + "," + std::to_string(slot_)
                    : prefix;

                return SendCommand(command);
            }
            return DEVICE_OK;
        })
    );
}
