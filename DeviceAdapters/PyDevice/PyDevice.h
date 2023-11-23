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

class CPyHub;

class CPyDeviceBase {
protected:
    bool initialized_ = false;
    PyObj object_;    
    string id_;
    CPyDeviceBase(const string& id) : id_(id), object_() {}
public:
    int CheckError() noexcept;
    PyObj Object() const {
        return object_;
    }
    vector<PyAction*> EnumerateProperties() noexcept;
protected:
    virtual void LogError(const char*) = 0;
};


/**
 * Base class for device adapters that are implement by a Python script.
 * Note: the MM API uses the Curiously Recurring Template pattern (antipattern). This strongly complicates everything. This is the reason for the class parameter T, the 'this->' prefixes, and the fact that all methods are declared in the header file.  
 * @tparam T Base type to implement. Should be CCameraBase, CGenericDevice, etc.
*/
template <class BaseType>
class CPyDeviceTemplate : public BaseType, public CPyDeviceBase
{
protected:

public:
    /**
     * Constructs a new device
     * The device is not initialized, and no Python calls are made. This only sets up error messages, the error handler, and three 'pre-init' properties that hold the path of the Python libraries, the path of the Python script, and the name of the Python class that implements the device.
     * @param adapterName name of the adapter type, e.g. "Camera". This is the default name of the Python class. For use by MM (GetName), the adapterName is prefixed with "Py", 
    */
    CPyDeviceTemplate(const string& id) : BaseType(), CPyDeviceBase(id)
    {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
        this->SetErrorText(ERR_PYTHON_DEVICE_NOT_FOUND, "");
    }
    virtual ~CPyDeviceTemplate() {}

    int CreateProperties(vector<PyAction*>& propertyDescriptors) noexcept {
        for (auto property: propertyDescriptors) {
            this->CreateProperty(property->name.c_str(), "", property->type, property->readonly, property, false);

            // Set limits. Only supported by MM if both upper and lower limit are present.
            if (property->has_limits)
                this->SetPropertyLimits(property->name.c_str(), property->min, property->max);
            

            // For enum-type objects (may be string, int or float), notify MM about the allowed values
            if (!property->enum_keys.empty())
                this->SetAllowedValues(property->name.c_str(), property->enum_keys);
        }
        propertyDescriptors.clear(); // remove the pointers from the vector because we transfered ownership of the Action objects in CreateProperty

        if (object_.HasAttribute("__doc__")) {
            auto doc = object_.Get("__doc__");
            if (doc != Py_None)
                this->SetDescription(doc.as<string>().c_str());
        }
        return DEVICE_OK;
    }
    
    /**
    * Checks if a Python error has occurred since the last call to CheckError
    * @return DEVICE_OK or ERR_PYTHON_EXCEPTION
    */
    void LogError(const char* err) {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, err);
        this->LogMessage(err);
    }

    /**
     * Executes the Python script and creates a Python object corresponding to the device
     * Initializes the Python interpreter (if needed).
     * The Python class may perform hardware initialization in its __init__ function. After creating the Python object and initializing it, the function 'InitializeDevice' is called, which may be overridden e.g. to check if all required properties are present on the Python object (see PyCamera for an example).
     * @return MM error code 
    */
    int Initialize() override {
        if (!initialized_) {
            object_ = CPyHub::GetDevice(id_);
            if (!object_) {
                this->SetErrorText(ERR_PYTHON_DEVICE_NOT_FOUND, ("Could not find the Python device id " + id_ + ". It may be that the Python script or the device object within it was renamed.").c_str());
                return ERR_PYTHON_DEVICE_NOT_FOUND;
            }

            auto propertyDescriptors = EnumerateProperties();
            _check_(CheckError());
            CreateProperties(propertyDescriptors);
            _check_(this->UpdateStatus()); // load value of all properties from the Python object
            initialized_ = true;
        }
        return DEVICE_OK;
    }

    /**
     * Destroys the Python object
     * @todo Currently, the Python interperter is nver de-initialized, even if all devices have been destroyed.
    */
    int Shutdown() override {
        object_.Clear();
        initialized_ = false;
        return DEVICE_OK;
    }

    void GetName(char* name) const override {
        CDeviceUtils::CopyLimitedString(name, id_.c_str());
    }
    bool Busy() override {
        return false;
    }
    
protected:
    CPyDeviceTemplate(CPyDeviceTemplate& other) = delete; // disable copy
};

/**
 * Class representing a generic device that is implemented by Python code
 * @todo add buttons to the GUI so that we can activate the device so that it actually does something
*/
using PyGenericClass = CPyDeviceTemplate<CGenericBase<std::monostate>>;
class CPyGenericDevice : public PyGenericClass {
public:
    CPyGenericDevice(const string& id) :PyGenericClass(id) {}
};


using PyHubClass = CPyDeviceTemplate<HubBase<std::monostate>>;

/**
   @brief Entry point for pydevice. This is the device that is listed in the hardware configuration manager.
   
   Only a single hub can be active at a time. Therefore, all devices that will be used in the configuration need to be constructed and
   initialized by a single Python script.
   
   Shutting down and initializing the Python runtime again is undefined behavior by the Python C API documentation. 
   Therefore, the runtime is only initialized the first time a Hub is initialized, and never de-initialized (see PyObj).
*/
class CPyHub : public PyHubClass {
    static constexpr const char* p_PythonScriptPath = "ScriptPath";
    static constexpr const char* p_PythonModulePath = "ModulePath";
public:
    static constexpr const char* g_adapterName = "PyHub";
    CPyHub();
    int Initialize() override;
    int Shutdown() override;

    static PyObj GetDevice(const string& device_id) noexcept;
    static bool SplitId(const string& id, string& deviceType, string& deviceName) noexcept;
    static string ComposeId(const string& deviceType, const string& deviceName) noexcept;

protected:
    int DetectInstalledDevices() override;
private:
    string LoadScript() noexcept;
    string ComputeModulePath() noexcept;
    
    // list of devices read from the `devices` dictionary that the Python script returns.
    std::map<string, PyObj> devices_;
    PyObj locals_;
    PyObj module_;

    // Location of the currently loaded script
    fs::path script_path_;

    // Pointer to the current (only) active Hub, or nullptr if no Hub is active.
    static CPyHub* g_the_hub;
};


