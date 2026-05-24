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
// AUTHOR:        Vikram Kopuri, based on Code by Nenad Amodaj Nico Stuurman and Jizhen Zhao
//

#pragma once

#include <string>

#include "DeviceBase.h"

class WPTRobot : public CGenericBase<WPTRobot> {
public:
    WPTRobot();
    ~WPTRobot();

    // MMDevice API
    bool Busy();
    void GetName(char* name) const;
    unsigned long GetNumberOfPositions() const { return 0; }

    int Initialize();
    int Shutdown();

private:
    // pre-init
    void CreatePortProperty();
    // properties
    void CreateStageProperty();
    void CreateSlotProperty();
    void CreateCommandProperty();

    bool initialized_ = false;
    long stage_ = 1;
    long slot_ = 1;

    // serial communication
    std::string port_;
    std::string command_;
};
