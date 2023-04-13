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

#define _check_(expression) { auto result = expression; if (result != DEVICE_OK) return result; }
// Base class for device adapters that are implement by a Python script.
// 
class CPyDeviceBase : public CGenericBase<CPyDeviceBase>
{
protected:
    PythonBridge _python;
    string _name;
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
    static constexpr const char* p_PythonDeviceClass = "Device class";
public:
    CPyDeviceBase(const string& name, unsigned int count) : _python(), _name(count > 0 ? name + std::to_string(count) : name) {
        SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not find python3.dll at the specified Python library path");
        SetErrorText(ERR_PYTHON_PATH_CONFLICT, "All Python devices must have the same Python library path");
        SetErrorText(ERR_PYTHON_SCRIPT_NOT_FOUND, "Could not find the python script at the specified location");
        SetErrorText(ERR_PYTHON_CLASS_NOT_FOUND, "Could not find a class definition with the specified name");
        SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
        SetErrorText(ERR_PYTHON_NO_INFO, "A Python error occurred, but no further information was available");

        // Adds properties for locating the Python libraries, the Python script, and the name of the device class
        CreateStringProperty(p_PythonHome, PythonBridge::FindPython().generic_string().c_str(), PythonBridge::PythonActive(), nullptr, true);
        CreateStringProperty(p_PythonScript, "", false, nullptr, true);
        CreateStringProperty(p_PythonDeviceClass, "Device", false, nullptr, true);
    }
    ~CPyDeviceBase() {}
    virtual void SetCallback(MM::Core* cbk) { 
        CGenericBase<CPyDeviceBase>::SetCallback(cbk);
        _python.SetErrorCallback([=](const string& message) { cbk->LogMessage(this, message.c_str(), false); });
    }

    // MMDevice API
    // ------------
    int Initialize() {
        // read the name of the python script and the name of the class representing the device. Pass these to the Python bridge
        // It would be much, much easier to directly access the _properties field, but that is made private.
        char pythonHome[MM::MaxStrLength] = { 0 };
        char pythonScript[MM::MaxStrLength] = { 0 };
        char pythonDeviceClass[MM::MaxStrLength] = { 0 };
        _check_(GetProperty(p_PythonHome, pythonHome));
        _check_(GetProperty(p_PythonScript, pythonScript));
        _check_(GetProperty(p_PythonDeviceClass, pythonDeviceClass));
        _check_(_python.Construct(pythonHome, pythonScript, pythonDeviceClass));

        for (const auto& option : _python.EnumerateProperties()) {
            switch (option.type) {
            case MM::String:
                this->CreateStringProperty(option.name.c_str(), "", false, new CPropertyAction(this, &CPyDeviceBase::OnString));
                break;
            case MM::Integer:
                this->CreateIntegerProperty(option.name.c_str(), 0, false, new CPropertyAction(this, &CPyDeviceBase::OnInteger));
                break;
            case MM::Float:
                this->CreateFloatProperty(option.name.c_str(), 0.0, false, new CPropertyAction(this, &CPyDeviceBase::OnFloat));
                break;
            }
        }
        return DEVICE_OK;
    }

    int OnString(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        /*if (eAct == MM::BeforeGet)
        {
            // nothing to do, let the caller use cached property
        }
        else if (eAct == MM::AfterSet)
        {
            double volts;
            pProp->Get(volts);
            return SetSignal(volts);
        }*/

        return DEVICE_OK;
    }

    int OnFloat(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        return DEVICE_OK;
    }
    int OnInteger(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        return DEVICE_OK;
    }

    int Shutdown() {
        return _python.Destruct();
    }
    void GetName(char* name) const {
        CDeviceUtils::CopyLimitedString(name, _name.c_str());
    }


protected:
};

class CPyGenericDevice : public CPyDeviceBase {
    static int g_GenericDeviceCount;
    static constexpr const char* g_AdapterName = "Generic Python device";
public:
    CPyGenericDevice() : CPyDeviceBase(g_AdapterName, g_GenericDeviceCount++) {}
    virtual bool Busy() { return false; }
};
#endif //_Pydevice_H_
