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
#include "actions.h"

class CPyHub;


/*
* A device id is of the form{ DeviceType }:{HubId} : {DeviceName}, where :
*  {DeviceType} is the device type : "Device", "Camera", etc.
*  {HubId} is the name of the script that was used to construct the device
*  {DeviceName} is the key of the 'devices' dictionary that contains the object
*/
inline bool split_id(const string& id, string& deviceType, string& hubId, string& deviceName) {
    auto colon1 = id.find(':');
    auto colon2 = id.find(':', colon1 + 1);
    if (colon1 != string::npos && colon2 != string::npos) {
        deviceType = id.substr(0, colon1);
        hubId = id.substr(colon1 + 1, colon2 - colon1 - 1);
        deviceName = id.substr(colon2 + 1);
        return true;
    } else
        return false;
};


/**
 * @brief Base class for all PyDevices, hold all common code that does not need call MM functions
*/
class CPyDeviceBase {
protected:
    /** Handle to the Python object */
    bool initialized_ = false;
    PyObj object_;    
    string id_;
    CPyHub* hub_ = nullptr;
    const function<void(const char*)> errorCallback_; // workaround for template madness

    int EnumerateProperties(vector<PyAction*>& propertyDescriptors) noexcept;
    CPyDeviceBase(const function<void(const char*)>& errorCallback, const string& id) : errorCallback_(errorCallback), object_(), id_(id) {
    }
    int CheckError() const noexcept;
};


/**
 * Base class for device adapters that are implement by a Python script.
 * Note: the MM API uses the Curiously Recurring Template pattern (antipattern). This strongly complicates everything. This is the reason for the class parameter T, the 'this->' prefixes, and the fact that all methods are declared in the header file.  
 * @tparam T Base type to implement. Should be CCameraBase, CGenericDevice, etc.
*/
template <template<class> class BaseType>
class CPyDeviceTemplate : public BaseType<CPyDeviceBase>, public CPyDeviceBase
{
protected:
public:
    /**
     * Constructs a new device
     * The device is not initialized, and no Python calls are made. This only sets up error messages, the error handler, and three 'pre-init' properties that hold the path of the Python libraries, the path of the Python script, and the name of the Python class that implements the device.
     * @param adapterName name of the adapter type, e.g. "Camera". This is the default name of the Python class. For use by MM (GetName), the adapterName is prefixed with "Py", 
    */
    CPyDeviceTemplate(const string& id) : CPyDeviceBase([this](const char* message) {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, message);
        this->SetErrorText(ERR_PYTHON_EXCEPTION, PyObj::g_errorMessage.c_str());
        }, id)
    {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
    }
    virtual ~CPyDeviceTemplate() {
    }
    int CreateProperties(vector<PyAction*>& propertyDescriptors) noexcept {
        for (auto property: propertyDescriptors) { // note: should be const auto&, but that does not work because a wrong signature in 'SetAllowedValues' (missing const)
            this->CreateProperty(property->name.c_str(), "", property->type, property->readOnly_, property, false);

            // Set limits. Only supported by MM if both upper and lower limit are present.
            if (property->has_limits)
                this->SetPropertyLimits(property->name.c_str(), property->min, property->max);
            

            // For enum-type objects (may be string, int or float), notify MM about the allowed values
            if (!property->enum_keys.empty())
                this->SetAllowedValues(property->name.c_str(), property->enum_keys);
        }
        propertyDescriptors.clear(); // remove the pointers from the vector because we transfered ownership of the Action objects in CreateProperty
        return DEVICE_OK;
    }

    /**
     * Executes the Python script and creates a Python object corresponding to the device
     * Initializes the Python interpreter (if needed).
     * The Python class may perform hardware initialization in its __init__ function. After creating the Python object and initializing it, the function 'InitializeDevice' is called, which may be overridden e.g. to check if all required properties are present on the Python object (see PyCamera for an example).
     * @return MM error code 
    */
    int Initialize() override {
        if (!initialized_) {
            // Locate parent hub 
            hub_ = static_cast<CPyHub*>(this->GetParentHub());
            if (!hub_)
                return DEVICE_COMM_HUB_MISSING;

            object_ = CPyHub::GetDevice(id_);
            if (!object_)
                return DEVICE_ERR;

            vector<PyAction*> propertyDescriptors;
            _check_(EnumerateProperties(propertyDescriptors));
            this->CreateProperties(propertyDescriptors);
            this->UpdateStatus(); // load value of all properties from the Python object
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
    
protected:
    // disable copy
    CPyDeviceTemplate(CPyDeviceTemplate& other) = delete;
};

/**
 * Class representing a generic device that is implemented by Python code
 * @todo add buttons to the GUI so that we can activate the device so that it actually does something
*/
using PyGenericClass = CPyDeviceTemplate<CGenericBase>;
class CPyGenericDevice : public PyGenericClass {
public:
    CPyGenericDevice(const string& id) :PyGenericClass(id) {
    }
    virtual bool Busy() override {
        return false;
    }
};

using PyCameraClass = CPyDeviceTemplate<CCameraBase>;
class CPyCamera : public PyCameraClass {
    /** numpy array corresponding to the last image, we hold a reference count so that we are sure the array does not get deleted during processing */
    PyObj lastImage_;       
    
    /** 'trigger' function of the camera object */
    PyObj triggerFunction_;
    PyObj readFunction_;    // 'wait' function of the camera object
    
public:
    CPyCamera(const string& id) : PyCameraClass(id) {
    }
    const unsigned char* GetImageBuffer() override;
    unsigned GetImageWidth() const override;
    unsigned GetImageHeight() const override;
    unsigned GetImageBytesPerPixel() const override;
    unsigned GetBitDepth() const override;
    long GetImageBufferSize() const override;
    int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) override;
    int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize) override;
    int ClearROI() override;
    double GetExposure() const override;
    void SetExposure(double exp) override;
    int GetBinning() const override;
    int SetBinning(int binF) override;
    int IsExposureSequenceable(bool& isSequenceable) const override;
    int SnapImage() override;
    int Shutdown() override;
    int Initialize() override;
};

struct DeviceDescriptor {
    PyObj object;
    string type;
};

using PyHubClass = CPyDeviceTemplate<HubBase>;
class CPyHub : public PyHubClass {
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
public:
    static constexpr const char* g_adapterName = "PyHub";
    CPyHub() : PyHubClass(g_adapterName) {
        SetErrorText(ERR_PYTHON_NOT_FOUND, "Could not initialize Python interpreter, perhaps an incorrect path was specified?");
        CreateStringProperty(p_PythonHome, PyObj::FindPython().generic_string().c_str(), false, nullptr, true);
        CreateStringProperty(p_PythonScript, "", false, nullptr, true);
    }
    virtual bool Busy() override {
        return false;
    }
    void GetName(char* name) const override {
        CDeviceUtils::CopyLimitedString(name, g_adapterName);
    }
    int Initialize() override;
    int Shutdown() noexcept;

    static PyObj GetDevice(const string& device_id) noexcept;

protected:
    int DetectInstalledDevices() override;
    int InitializeInterpreter() noexcept;
    int RunScript() noexcept;
private:
    bool initialized_ = false;

    // Global interpreter lock (GIL) for the Python interpreter. Before doing anything Python, we need to obtain the GIL
    // Note that all CPyHub's share the same interpreter
    static PyThreadState* g_threadState;
    std::map<string, PyObj> devices_;

    // List of all Hub objects currently in existence. todo: when this number drops to 0, the Python interpreter is destroyed
    static std::map<string, CPyHub*> g_hubs;
};


