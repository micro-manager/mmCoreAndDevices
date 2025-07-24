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

#include <sstream>
#include <string>

#include "DeviceUtils.h"
#include "ModuleInterface.h"

const char* g_WPTRobotName = "WPTRobot";

// Exported MMDevice API
MODULE_API void InitializeModuleData() {
    RegisterDevice(g_WPTRobotName, MM::GenericDevice, "WPTRobot");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName) {
    if (deviceName == nullptr) {
        return nullptr;
    }

    if (std::string(deviceName) == g_WPTRobotName) {
        return new WPTRobot();
    }

    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
    delete pDevice;
}

// Clear contents of serial port
int ClearPort(MM::Device& device, MM::Core& core, const std::string& port) {
    constexpr size_t bufferSize = 255;
    unsigned char clear[bufferSize];
    unsigned long read = bufferSize;
    int result;
    while (read == bufferSize) {
        result = core.ReadFromSerial(&device, port.c_str(), clear, bufferSize, read);
        if (result != DEVICE_OK) {
            return result;
        }
    }
    return DEVICE_OK;
}

WPTRobot::WPTRobot() :
    CGenericBase<WPTRobot>(),
    initialized_(false),
    numPos_(0),
    port_(""),
    command_(""),
    stage_(1),
    slot_(1) {

    InitializeDefaultErrorMessages();

    // Create Preinitialization Properties

    // Name
    CreateProperty(MM::g_Keyword_Name, g_WPTRobotName, MM::String, true);

    // Description
    CreateProperty(MM::g_Keyword_Description, "Wellplate Transfer Robot", MM::String, true);

    // Port
    CPropertyAction* pAct = new CPropertyAction(this, &WPTRobot::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

WPTRobot::~WPTRobot() {
    Shutdown();
}

void WPTRobot::GetName(char* name) const {
    CDeviceUtils::CopyLimitedString(name, g_WPTRobotName);
}

int WPTRobot::Initialize() {
    // empty the Rx serial buffer before sending command
    ClearPort(*this, *GetCoreCallback(), port_.c_str());

    // Stage and Slot passed as properties
    CPropertyAction* pAct = new CPropertyAction(this, &WPTRobot::OnStage);
    CreateProperty("Stage", "1", MM::Integer, false, pAct);
    // SetPropertyLimits("Stage", 1, 3); // setting limits must be edited again, if set up changed

    pAct = new CPropertyAction(this, &WPTRobot::OnSlot);
    CreateProperty("Slot", "1", MM::Integer, false, pAct);
    // SetPropertyLimits("Slot", 1, 10);

    // setting this property to different keywords leads to a command
    pAct = new CPropertyAction(this, &WPTRobot::OnCommand);
    CreateProperty("Command", "Undefined", MM::String, false, pAct);

    int ret = UpdateStatus();
    if (ret != DEVICE_OK) {
        return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int WPTRobot::Shutdown() {
    if (initialized_) {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool WPTRobot::Busy() {
    return false; // reply is only given after move is completed, so Busy() not necessary
}

int WPTRobot::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(port_.c_str());
    } else if (eAct == MM::AfterSet) {
        if (initialized_) {
            // revert
            pProp->Set(port_.c_str());
            return ERR_PORT_CHANGE_FORBIDDEN;
        }
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

// Just reading and writing to the stage variable
int WPTRobot::OnStage(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(stage_);
    } else if (eAct == MM::AfterSet) {
        pProp->Get(stage_);
    }
    return DEVICE_OK;
}

// Just reading and writing to the slot variable
int WPTRobot::OnSlot(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(slot_);
    } else if (eAct == MM::AfterSet) {
        pProp->Get(slot_);
    }
    return DEVICE_OK;
}

int WPTRobot::OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(command_.c_str());
    } else if (eAct == MM::AfterSet) {
        // Read what keyword the user issued, and send the corresponding command
        pProp->Get(command_);

        std::ostringstream os;
        std::string answer;
        int ret;

        if (command_.compare(0, 3, "ORG") == 0) {
            // user issued ORG command
            os << "ORG";

            ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r\n");
            if (ret != DEVICE_OK) {
                return ret;
            }

            ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
            if (ret != DEVICE_OK) {
                return ret;
            }

            // checking if the answer is what is expected,
            // if not just give an error and quit
            if (answer.length() != 3) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
            if (answer.compare(0, 3, "ORG") != 0) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
        } else if (command_.compare(0, 3, "GET") == 0) {
            // user issued GET command
            os << "GET " << stage_ << "," << slot_;

            ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r\n");
            if (ret != DEVICE_OK) {
                return ret;
            }

            ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
            if (ret != DEVICE_OK) {
                return ret;
            }

            if (answer.length() != 3) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
            if (answer.compare(0, 3, "GET") != 0) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
        } else if (command_.substr(0, 3) == "PUT") {
            // user issued PUT command
            os << "PUT " << stage_ << "," << slot_;

            ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r\n");
            if (ret != DEVICE_OK) {
                return ret;
            }

            ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
            if (ret != DEVICE_OK) {
                return ret;
            }

            if (answer.length() != 3) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
            if (answer.compare(0, 3, "PUT") != 0) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
        } else if (command_.compare(0, 3, "AES") == 0) {
            // used issued STOP command

            // AES is command for emergency stop, stops the robot cold, issue DRT command to enable again
            os << "AES";

            ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r\n");
            if (ret != DEVICE_OK) {
                return ret;
            }

            ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
            if (ret != DEVICE_OK) {
                return ret;
            }

            if (answer.length() != 3) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
            if (answer.compare(0, 3, "AES") != 0) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
        } else if (command_.compare(0, 3, "DRT") == 0) {
            // user issued DRT command

            // override errors
            os << "DRT";

            ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r\n");
            if (ret != DEVICE_OK) {
                return ret;
            }

            ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
            if (ret != DEVICE_OK) {
                return ret;
            }

            if (answer.length() != 3) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
            if (answer.compare(0, 3, "DRT") != 0) {
                return ERR_UNRECOGNIZED_ANSWER;
            }
        }
    }
    return DEVICE_OK;
}
