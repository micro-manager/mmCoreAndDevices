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
#pragma once
#include "pch.h"
#include "PyObj.h"
#include "Actions.h"

const char* const g_Method_Busy = "busy";

class CPyHub;
tuple<vector<PyAction*>, PyObj> EnumerateProperties(const PyObj& deviceInfo, const ErrorCallback& callback) noexcept;

/**
 * Base class for device adapters that are implement by a Python script.
 * @tparam BaseType Base type to implement. Should be CCameraBase, CGenericDevice, etc.
*/
template <class BaseType>
class CPyDeviceTemplate : public BaseType
{
protected:
    bool initialized_ = false;
    PyObj busy_; // busy() method
    string id_;

public:
    /**
     * Constructs a new device
     * The device is not initialized, and no Python calls are made. This only sets up error messages, the error handler, and three 'pre-init' properties that hold the path of the Python libraries, the path of the Python script, and the name of the Python class that implements the device.
     * @param id device type, e.g. "Camera:cam". 
    */
    CPyDeviceTemplate(const string& id) : BaseType(), id_(id)
    {
        this->SetErrorText(
            ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
        this->SetErrorText(ERR_PYTHON_RUNTIME_NOT_FOUND, "");
    }

    int CreateProperties(const vector<PyAction*>& propertyDescriptors) noexcept
    {
        for (auto property : propertyDescriptors)
        {
            this->CreateProperty(property->name.c_str(), "", property->type, property->readonly, property, false);

            // Set limits. Only supported by MM if both upper and lower limit are present.
            if (property->has_limits)
                this->SetPropertyLimits(property->name.c_str(), property->min, property->max);


            // For enum-type objects (may be string, int or float), notify MM about the allowed values
            if (!property->enum_keys.empty())
                this->SetAllowedValues(property->name.c_str(), property->enum_keys);
        }
        /*
        
        propertyDescriptors.clear(); // remove the pointers from the vector because we transfered ownership of the Action objects in CreateProperty
        
        if (deviceInfo.HasAttribute("__doc__")) {
            auto doc = deviceInfo.Get("__doc__");
            if (doc != Py_None)
                this->SetDescription(doc.as<string>().c_str());
        }*/
        return DEVICE_OK;
    }

    /**
    * Checks if a Python error has occurred since the last call to CheckError
    * @return DEVICE_OK or ERR_PYTHON_EXCEPTION
    */
    int CheckError() noexcept
    {
        PyLock lock;
        PyObj::ReportError(); // check if any new errors happened
        if (!PyObj::g_errorMessage.empty())
        {
            // note: thread safety of this part relies on the PyLock
            auto& err = PyObj::g_errorMessage;
            this->SetErrorText(ERR_PYTHON_EXCEPTION, err.c_str());
            this->LogMessage(err.c_str()); //note: is this function thread safe??
            PyObj::g_errorMessage.clear();
            return ERR_PYTHON_EXCEPTION;
        }
        return DEVICE_OK;
    }

    /**
     * Executes the Python script and creates a Python object corresponding to the device
     * Initializes the Python interpreter (if needed).
     * The Python class may perform hardware initialization in its __init__ function. After creating the Python object and initializing it, the function 'InitializeDevice' is called, which may be overridden e.g. to check if all required properties are present on the Python object (see PyCamera for an example).
     * @return MM error code 
    */
    int Initialize() override
    {
        if (!initialized_)
        {
            auto deviceInfo = CPyHub::GetDeviceInfo(id_);
            if (!deviceInfo)
            {
                string deviceType, deviceName;
                CPyHub::SplitId(id_, deviceType, deviceName);
                auto altId = CPyHub::ComposeId("Device",deviceName);
                deviceInfo = CPyHub::GetDeviceInfo(altId);
                if (!deviceInfo) {
                    this->SetErrorText(
                        ERR_PYTHON_RUNTIME_NOT_FOUND,
                        ("Could not find the Python device id " + id_ +
                            ". It may be that the Python script or the device object within it was renamed.").c_str());
                    return ERR_PYTHON_RUNTIME_NOT_FOUND;
                } else
                {
                    auto msg = "Did not recognize device type " + deviceType;
                    this->CreateProperty("WARNING", msg.c_str(), MM::String, true, nullptr, false);
                }
            }
            auto [properties, methods] = EnumerateProperties(deviceInfo, [this]() { return this->CheckError(); });
            _check_(CheckError());
            _check_(CreateProperties(properties));
            _check_(ConnectMethods(methods));
            _check_(this->UpdateStatus()); // load value of all properties from the Python object
            initialized_ = true;
        }
        return DEVICE_OK;
    }

    long GetLongProperty(const char* property) const
    {
        long value = 0;
        // Unfortunately, GetProperty is 'const' for some (historical?) reason.
        // Therefore, we need to manually remove the const qualifier from 'this'
        const_cast<BaseType*>(static_cast<const BaseType*>(this))->GetProperty(property, value);
        return value;
    }

    int SetLongProperty(const char* property, long value)
    {
        return this->SetProperty(property, std::to_string(value).c_str());
    }

    double GetFloatProperty(const char* property) const
    {
        double value = 0.0;
        const_cast<BaseType*>(static_cast<const BaseType*>(this))->GetProperty(property, value);
        return value;
    }

    int SetFloatProperty(const char* property, double value)
    {
        return this->SetProperty(property, std::to_string(value).c_str());
    }

    virtual int ConnectMethods(const PyObj& methods)
    {
        busy_ = methods.GetDictItem("busy");
        return CheckError();
    }

    /**
     * Destroys the Python object
     * @todo Currently, the Python interpreter is never de-initialized, even if all devices have been destroyed.
    */
    int Shutdown() override
    {
        initialized_ = false;
        return DEVICE_OK;
    }

    void GetName(char* name) const override
    {
        CDeviceUtils::CopyLimitedString(name, id_.c_str());
    }

    bool Busy() override
    {
        if (!busy_)
            return false; // device does not have a busy() method

        auto retval = busy_.Call().as<bool>();
        CheckError();
        return retval;
    }

    CPyDeviceTemplate(CPyDeviceTemplate& other) = delete; // disable copy
};

/**
 * Class representing a generic device that is implemented by Python code
 * @todo add buttons to the GUI so that we can activate the device so that it actually does something
*/
using PyGenericClass = CPyDeviceTemplate<CGenericBase<std::monostate>>;

class CPyGenericDevice : public PyGenericClass
{
public:
    CPyGenericDevice(const string& id) : PyGenericClass(id)
    {
    }
};


using PyHubClass = CPyDeviceTemplate<HubBase<std::monostate>>;

/**
   @brief Entry point for pydevice. This is the device that is listed in the hardware configuration manager.
   
   Only a single hub can be active at a time. Therefore, all devices that will be used in the configuration need to be constructed and
   initialized by a single Python script.
   
   Shutting down and initializing the Python runtime again is undefined behavior by the Python C API documentation. 
   Therefore, the runtime is only initialized the first time a Hub is initialized, and never de-initialized (see PyObj).
*/
class CPyHub : public PyHubClass
{
    static constexpr const char* p_PythonScriptPath = "ScriptPath";
    static constexpr const char* p_PythonPath = "PythonEnvironment";
    static constexpr const char* p_DllPath = "Python executable";
    static constexpr const char* p_VirtualEnvironment = "Virtual environment";

public:
    static constexpr const char* g_adapterName = "PyHub";
    CPyHub();
    int Initialize() override;
    int Shutdown() override;

    static PyObj GetDeviceInfo(const string& device_id) noexcept;
    static bool SplitId(const string& id, string& deviceType, string& deviceName) noexcept;
    static string ComposeId(const string& deviceType, const string& deviceName) noexcept;

protected:
    int DetectInstalledDevices() override;

private:
    string LoadScript() noexcept;

    // list of devices read from the `devices` dictionary that the Python script returns.
    std::map<string, PyObj> devices_;

    // Location of the currently loaded script
    fs::path script_path_;

    // Pointer to the current (only) active Hub, or nullptr if no Hub is active.
    static CPyHub* g_the_hub;
};
