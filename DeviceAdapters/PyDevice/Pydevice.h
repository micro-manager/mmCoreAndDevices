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

struct PropertyDescriptor {
    string name;
    int type;
    double min = 0.0;
    double max = 0.0;
    bool has_limits = false;
    vector<string> allowed_values;
};
class CPyHub;

/**
 * @brief Base class for all PyDevices, hold all common code that does not need call MM functions
*/
class CPyDeviceBase {
protected:
    /** Handle to the Python object */
    PyObj object_;    
    bool initialized_ = false;
    vector<PropertyDescriptor> propertyDescriptors_;
    const function<void(const char*)> errorCallback_; // workaround for template madness
    string name_;

    int EnumerateProperties(const CPyHub& hub) noexcept;
    CPyDeviceBase(const function<void(const char*)>& errorCallback, const string& name, const PyObj& object) : errorCallback_(errorCallback), object_(object), name_(name) {
    }


    template <class T> int Get(const PyObj& object, const char* name, T& value) const noexcept {
        PyLock lock;
        value = PyObj(PyObject_GetAttrString(object, name)).as<T>();
        return CheckError();
    }

    /**
    * Checks if a Python error has occurred since the last call to CheckError
    * @return DEVICE_OK or ERR_PYTHON_EXCEPTION
    */
    int CheckError() const noexcept {
        PyLock lock;
        PyObj::ReportError(); // check if any new errors happened
        if (!PyObj::g_errorMessage.empty()) {
            errorCallback_(PyObj::g_errorMessage.c_str());
            PyObj::g_errorMessage.clear();
            return ERR_PYTHON_EXCEPTION;
        }
        else
            return DEVICE_OK;
    }


public:
    int OnObjectProperty(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
     * Callback that is called when a property value is read or written
     * We respond to property writes by relaying the value to the Python object.
     * In this implementation, we don't perform any action when the value is read. We just use the cached value in MM.
     * Note that this is not correct if the property is changed by Python.
     * @return MM result code
    */
    template <class T> int OnProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
    {
        //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
        if (eAct == MM::AfterSet)
        {
            T value = {};
            pProp->Get(value);
            object_.Set(pProp->GetName().c_str(), value);
            return CheckError();
        }
        return DEVICE_OK;
    }
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
    CPyDeviceTemplate(const string& name, const PyObj& object) : CPyDeviceBase([this](const char* message) {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, message);
        this->SetErrorText(ERR_PYTHON_EXCEPTION, PyObj::g_errorMessage.c_str());
        }, name, object)
    {
        this->SetErrorText(ERR_PYTHON_EXCEPTION, "The Python code threw an exception, check the CoreLog error log for details");
    }
    virtual ~CPyDeviceTemplate() {
    }
    int CreateProperties() noexcept {
        using Action = typename MM::Action<CPyDeviceBase>;
        for (auto& property: propertyDescriptors_) { // note: should be const auto&, but that does not work because a wrong signature in 'SetAllowedValues' (missing const)
            switch (property.type) {
            case MM::Integer:
                _check_(this->CreateIntegerProperty(property.name.c_str(), 0, false, new Action(this, &CPyDeviceBase::OnProperty<long>)));
                break;
            case MM::Float:
                _check_(this->CreateFloatProperty(property.name.c_str(), 0.0, false, new Action(this, &CPyDeviceBase::OnProperty<double>)));
                break;
            case MM::String:
                _check_(this->CreateStringProperty(property.name.c_str(), "", false, new Action(this, &CPyDeviceBase::OnProperty<string>)));
                break;
            case MM::Undef:
                _check_(this->CreateStringProperty(property.name.c_str(), "", false, new Action(this, &CPyDeviceBase::OnObjectProperty)));
                break;
            };

            // Set limits. Only supported by MM if both upper and lower limit are present.
            // The min/max attributes are always present, we only need to check if they don't hold 'None'
            if (property.has_limits)
                this->SetPropertyLimits(property.name.c_str(), property.min, property.max);
            

            // For enum-type objects (may be string, int or float), notify MM about the allowed values
            // The allowed_values attribute is always present, we only need to check if they don't hold 'None'
            if (!property.allowed_values.empty())
                this->SetAllowedValues(property.name.c_str(), property.allowed_values);
        }
        return CheckError();
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
            auto hub = static_cast<CPyHub*>(this->GetParentHub());
            if (!hub)
                return DEVICE_COMM_HUB_MISSING;

            // for backward comp (??)
            char hubLabel[MM::MaxStrLength];
            hub->GetLabel(hubLabel);
            this->SetParentID(hubLabel); 

            if (object_) {
                EnumerateProperties(*hub);
                this->CreateProperties();
            }
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

    /**
     * Required by MM::Device API. Returns name of the adapter type
    */
    void GetName(char* name) const override {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }
    
protected:
    /**
    * Called after construction of the Python class
    * May be overridden by a derived class to check if all required properties are present and have the correct type,
    * or to perform any other initialization if needed.
    */
    virtual int InitializeDevice() {
        return DEVICE_OK;
    }
    
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
    CPyGenericDevice(const string& name, const PyObj& object) :PyGenericClass(name, object) {
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
    PyObj waitFunction_;    // 'wait' function of the camera object
    
public:
    CPyCamera(const string& name, const PyObj& object) : PyCameraClass(name, object) {
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
protected:
    int InitializeDevice() override;
};

using PyHubClass = CPyDeviceTemplate<HubBase>;
class CPyHub : public PyHubClass {
    static constexpr const char* p_PythonHome = "Python library path";
    static constexpr const char* p_PythonScript = "Device script";
public:
    static constexpr const char* g_adapterName = "PyHub";
    CPyHub() : PyHubClass(g_adapterName, PyObj()) {
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

public: // todo: will be obsolete
    PyObj intPropertyType_;
    PyObj floatPropertyType_;
    PyObj stringPropertyType_;
    PyObj objectPropertyType_;
};