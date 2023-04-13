///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic device adapter that runs a Python script. Serves as base class for PyCamera, etc.
//                
// AUTHOR:        Ivo Vellekoop
//                Jeroen Doornbos
//
// COPYRIGHT:     University of Twente, Enschede, The Netherlands.
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef _Pydevice_H_
#define _Pydevice_H_

#include <string>
#include "DeviceBase.h"
#include "PythonBridge.h"

// Base class for device adapters that are implement by a Python script.
// 
class CPyGenericDevice : public CGenericBase<PythonBridge>
{
protected:
    PythonBridge _python;
    static constexpr const char* g_AdapterName = "Generic Python device";
public:
    CPyGenericDevice() : _python() {
        // note: this code duplication is needed because SetErrorText is not public
        SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not find python3.dll at the specified Python library path");
        SetErrorText(ERR_PYTHON_PATH_CONFLICT, "All Python devices must have the same Python library path");
        SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the python script at the specified location");
        SetErrorText(ERR_PYTHON_CLASS_NOT_FOUND, "Could not find a class definition with the specified name");
        SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
        SetErrorText(ERR_PYTHON_NO_INFO, "A Python error occurred, but no further information was available");
        
        _python.Construct(this);
    }
    ~CPyGenericDevice() {
    }
    int Initialize() {
        return _python.Initialize(this, GetCoreCallback());
    }
    int Shutdown() {
        return _python.Destruct();
    }
    void GetName(char* name) const {
        CDeviceUtils::CopyLimitedString(name, g_AdapterName);
    }
    virtual bool Busy() { return false; }


protected:
};

#endif //_Pydevice_H_
