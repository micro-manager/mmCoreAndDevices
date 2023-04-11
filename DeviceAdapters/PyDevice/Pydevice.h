///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic device adapter that runs a Python script. Serves as base class for PyCamera, etc.
//                
// AUTHOR:        Jeroen Doornbos
//                Ivo Vellekoop
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

#define _check_(expression) { auto result = expression; if (result != DEVICE_OK) return result; }
// Base class for device adapters that are implement by a Python script.
// 
template <class BaseClass>
class CPyDeviceBase : public BaseClass
{
protected:
    PythonBridge _python;
    string _name;
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
    static constexpr const char* p_PythonDeviceClass = "Device class";
public:
    CPyDeviceBase(const string& name, unsigned int count) : _python(), _name(count > 0 ? name + std::to_string(count) : name) {
        this->SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not find python3.dll at the specified Python library path");
        this->SetErrorText(ERR_PYTHON_PATH_CONFLICT, "All Python devices must have the same Python library path");
        this->SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the python script at the specified location");
        this->SetErrorText(ERR_PYTHON_CLASS_NOT_FOUND, "Could not find a class definition with the specified name");
        this->SetErrorText(ERR_BOOTSTRAP_COMPILATION_FAILED, "Could not compile Python loader script");

        // Adds properties for locating the Python libraries, the Python script, and the name of the device class
        this->CreateStringProperty(p_PythonHome, PythonBridge::FindPython().c_str(), PythonBridge::PythonActive(), nullptr, true);
        this->CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        this->CreateStringProperty(p_PythonDeviceClass, "Device", false, nullptr, true);
    }
    ~CPyDeviceBase() {}

    // MMDevice API
    // ------------
    int Initialize() {
        // read the name of the python script and the name of the class representing the device. Pass these to the Python bridge
        // It would be much, much easier to directly access the _properties field, but that is made private.
        char pythonHome[MM::MaxStrLength] = { 0 };
        char pythonScript[MM::MaxStrLength] = { 0 };
        char pythonDeviceClass[MM::MaxStrLength] = { 0 };
        _check_(this->GetProperty(p_PythonHome, pythonHome));
        _check_(this->GetProperty(p_PythonScript, pythonScript));
        _check_(this->GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(_python.Construct(pythonHome, pythonScript, pythonDeviceClass));
        return DEVICE_OK;
//        for (const auto& option : _python.Options()) {
//            this->CreateProperty(option.GetName(), option.GetValueString(), option.GetType(), false, new CPropertyAction(_python, &_python::OnPropertyAccess);
//        }
//        MM::MaxStrLength
    }
    int Shutdown() {
        return _python.Destruct();
    }
    void GetName(char* name) const {
        CDeviceUtils::CopyLimitedString(name, _name.c_str());
    }


protected:
};

class CPyGenericDevice : public CPyDeviceBase<CGenericBase<PythonBridge>> {
    static int g_GenericDeviceCount;
    static constexpr const char* g_AdapterName = "Generic Python device";
public:
    CPyGenericDevice() : CPyDeviceBase<CGenericBase<PythonBridge>>(g_AdapterName, g_GenericDeviceCount++) {}
    virtual bool Busy() { return false; }
};
#endif //_Pydevice_H_
