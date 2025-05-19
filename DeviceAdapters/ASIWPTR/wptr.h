///////////////////////////////////////////////////////////////////////////////
// FILE:          WPTR.h
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
// AUTHOR:         Vikram Kopuri, based on Code by Nenad Amodaj Nico Stuurman and Jizhen Zhao
//

#ifndef ASIWPTR_H
#define ASIWPTR_H

#include <string>

#include "DeviceBase.h"
#include "MMDevice.h"

// Error codes
//#define ERR_UNKNOWN_POSITION         10002
#define ERR_PORT_CHANGE_FORBIDDEN    10004
#define ERR_INVALID_STEP_SIZE        10006
#define ERR_INVALID_MODE             10008
#define ERR_UNRECOGNIZED_ANSWER      10009
#define ERR_UNSPECIFIED_ERROR        10010

#define ERR_OFFSET 10100

int ClearPort(MM::Device& device, MM::Core& core, std::string port);

class WPTRobot : public CGenericBase<WPTRobot> {
public:
    WPTRobot();
    ~WPTRobot();

    // MMDevice API
    bool Busy();
    void GetName(char* name) const;
    unsigned long GetNumberOfPositions() const { return numPos_; }

    int Initialize();
    int Shutdown();

    int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnStage(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSlot(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    bool initialized_;
    unsigned int numPos_;
    std::string port_;    // MMCore name of serial port
    std::string command_; // Command exchange with MMCore
    long stage_;
    long slot_;
};

#endif // ASIWPTR_H
