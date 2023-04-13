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
template <class T>
class CPyDeviceBase : public T
{
protected:
    const char* _adapterName;
    PythonBridge _python;
public:
    CPyDeviceBase(const char* adapterName) : _adapterName(adapterName), _python([this](const char* message) { this->LogMessage(message, false); }) {
        this->SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not find python3.dll at the specified Python library path");
        this->SetErrorText(ERR_PYTHON_PATH_CONFLICT, "All Python devices must have the same Python library path");
        this->SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the python script at the specified location");
        this->SetErrorText(ERR_PYTHON_CLASS_NOT_FOUND, "Could not find a class definition with the specified name");
        this->SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
        this->SetErrorText(ERR_PYTHON_NO_INFO, "A Python error occurred, but no further information was available");
        
        _python.Construct(this);
    }
    ~CPyDeviceBase() {
    }
    int Initialize() {
        return _python.Initialize(this);
    }
    int Shutdown() {
        return _python.Destruct();
    }
    void GetName(char* name) const {
        CDeviceUtils::CopyLimitedString(name, _adapterName);
    }
    virtual bool Busy() { return false; }
};

class CPyGenericDevice : public CPyDeviceBase<CGenericBase<PythonBridge>> {
public:
    CPyGenericDevice() : CPyDeviceBase<CGenericBase<PythonBridge>>("Generic Python device") {

    }

};
#endif //_Pydevice_H_
