///////////////////////////////////////////////////////////////////////////////
// FILE:          Pydevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The implementation of the Python camera. Adapted from the Democamera in the 
//                Micromanager repository.
//                
// AUTHOR:        Jeroen Doornbos
//                Ivo Vellekoop
// COPYRIGHT:     
// LICENSE:       ?
#include "pch.h"
#include "PyDevice.h"
#include <numpy/arrayobject.h>

PyThreadState* CPyHub::g_threadState = nullptr;
std::map<string, CPyHub*> CPyHub::g_hubs;


/**
 * Destroys all Python objects by releasing the references we currently have
 * 
 * @return 
*/
int CPyHub::Shutdown() noexcept {
    PyLock lock;
    devices_.clear();
    intPropertyType_.Clear();
    floatPropertyType_.Clear();
    stringPropertyType_.Clear();
    objectPropertyType_.Clear();
    cameraProtocol_.Clear();
    initialized_ = false;

    g_hubs.erase(name_);
    return PyHubClass::Shutdown();
}

int CPyHub::DetectInstalledDevices() {
    ClearInstalledDevices();
    for (const auto& key_value : devices_) {
        // todo: find device type
        auto name = key_value.second.type + ":" + name_ + ':' + key_value.first;
        auto mm_device = new CPyGenericDevice(name);
        AddInstalledDevice(mm_device);
    }
    return CheckError();
}

/**
 * @brief Initialize the Python interpreter, run the script, and convert the 'devices' dictionary into a c++ map
*/
int CPyHub::Initialize() {
    if (!initialized_) {
        _check_(InitializeInterpreter());
        _check_(RunScript());
        initialized_ = true;
    }
    return CheckError();
}

/**
 * Initialize the Python interpreter
 * If a non-empty pythonHome path is specified, the Python install from that path is used, otherwise the Python API tries to find an interpreter itself. 
 * If a Python iterpreter is already running, the old one is used and the path is ignored
 * todo: can we use virtual environments?
*/
int CPyHub::InitializeInterpreter() noexcept
{
    // Initilialize Python interpreter, if not already done
    if (g_threadState == nullptr) {
        // Initialize Python configuration (new style)
        // The old style initialization (using Py_Initialize) does not have a way to report errors. In particular,
        // if the Python installation cannot be found, the program just terminates!
        // The new style initialization returns an error, that can then be shown to the user instead of crashing micro manager.
        PyConfig config;
        PyConfig_InitPythonConfig(&config);

        char pythonHomeString[MM::MaxStrLength] = { 0 };
        _check_(GetProperty(p_PythonHome, pythonHomeString));
        const fs::path& pythonHome(pythonHomeString);
        if (!pythonHome.empty())
            PyConfig_SetString(&config, &config.home, pythonHome.c_str());

        auto status = Py_InitializeFromConfig(&config);

        //PyConfig_Read(&config); // for debugging
        PyConfig_Clear(&config);
        if (PyStatus_Exception(status))
            return ERR_PYTHON_NOT_FOUND;
    
        _import_array(); // initialize numpy. We don't use import_array (without _) because it hides any error message that may occur.
    
        // allow multi threading and store the thread state (global interpreter lock).
        // note: savethread releases the lock.
        g_threadState = PyEval_SaveThread();
    }
    return CheckError();
}

// uses reflection to locate all accessible properties of the device object.
int CPyDeviceBase::EnumerateProperties(const CPyHub& hub) noexcept
{
    PyLock lock;
    propertyDescriptors_.clear();
    auto type_info = PyObj(PyObject_Type(object_));
    auto dict = PyObj(PyObject_GetAttrString(type_info, "__dict__"));
    auto properties = PyObj(PyMapping_Items(dict)); // key-value pairs
    auto property_count = PyList_Size(properties);
    for (Py_ssize_t i = 0; i < property_count; i++) {
        PropertyDescriptor descriptor;

        auto key_value = PyList_GetItem(properties, i); // note: borrowed reference, don't ref count (what a mess...)
        descriptor.name = PyObj::Borrow(PyTuple_GetItem(key_value, 0)).as<string>();
        auto property = PyObj::Borrow(PyTuple_GetItem(key_value, 1));

        if (PyObject_IsInstance(property, hub.intPropertyType_))
            descriptor.type = MM::Integer;
        else if (PyObject_IsInstance(property, hub.floatPropertyType_))
            descriptor.type = MM::Float;
        else if (PyObject_IsInstance(property, hub.stringPropertyType_))
            descriptor.type = MM::String;
        else if (PyObject_IsInstance(property, hub.objectPropertyType_))
            descriptor.type = MM::Undef;
        else
            continue;

        // Set limits. Only supported by MM if both upper and lower limit are present.
        // The min/max attributes are always present, we only need to check if they don't hold 'None'
        if (descriptor.type == MM::Integer || descriptor.type == MM::Float) {
            auto lower = property.Get("min");
            auto upper = property.Get("max");
            if (lower != Py_None && upper != Py_None) {
                descriptor.min = lower.as<double>();
                descriptor.max = upper.as<double>();
                descriptor.has_limits = true;
            }
        }

        // For enum-type objects (may be string, int or float), notify MM about the allowed values
        // The allowed_values attribute is always present, we only need to check if they don't hold 'None'
        PyObj allowed_values = property.Get("allowed_values");
        if (allowed_values != Py_None) {
            auto value_count = PyList_Size(allowed_values);
            for (Py_ssize_t j = 0; j < value_count; j++) {
                auto value = PyList_GetItem(allowed_values, j); // borrowed reference, don't ref count
                descriptor.allowed_values.push_back(PyObj(PyObject_Str(value)).as<string>());
            }
        }
        
        propertyDescriptors_.push_back(descriptor);
    }
    return CheckError();
}

/**
* Checks if a Python error has occurred since the last call to CheckError
* @return DEVICE_OK or ERR_PYTHON_EXCEPTION
*/
int CPyDeviceBase::CheckError() const noexcept {
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


int CPyDeviceBase::OnObjectProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    throw std::exception("not implemented");
    //if (eAct == MM::BeforeGet) // nothing to do, let the caller use cached property
   /* if (eAct == MM::AfterSet)
    {
        string label;
        pProp->Get(label);
        auto device = g_Devices.find(label); // look up device by name
        if (device != g_Devices.end()) {
            return SetProperty(pProp->GetName().c_str(), device->second);
        }
        else { // label not found. This could be because the object is not constructed yet
            g_MissingLinks.push_back({ object_, pProp->GetName(), label });
        }
    }
    return DEVICE_OK;*/
}

/**
 * Loads the Python script and creates a device object
 * @param pythonScript path of the .py script file
 * @param pythonClass name of the Python class to create an instance of
 * @return MM return code
*/
int CPyHub::RunScript() noexcept {
    PyLock lock;
    char scriptPathString[MM::MaxStrLength] = { 0 };
    _check_(GetProperty(p_PythonScript, scriptPathString));
    const fs::path& scriptPath(scriptPathString);

    auto bootstrap = std::stringstream();
    bootstrap << "SCRIPT_PATH = '" << scriptPath.parent_path().generic_string() << "'\n";
    bootstrap << "SCRIPT_FILE = '" << scriptPath.generic_string() << "'\n";
    bootstrap << R"raw(
import sys
sys.path.append(SCRIPT_PATH)
code = open(SCRIPT_FILE)
exec(code.read())
code.close()

import numpy as np
from typing import Protocol, runtime_checkable

@runtime_checkable
class MMCamera(Protocol):
    width: int
    height: int
    top: int
    left: int
    Binning: int
    exposure_ms: float
    image: np.ndarray
    def trigger():
        pass
    def wait():
        pass

def extract_metadata(obj):
    if isinstance(obj, MMCamera):
        type = "Camera"
    else:
        type = "Device"
    return (type, obj)
    
metadata = {name:extract_metadata(device) for name,device in devices.items()}
)raw";

    auto scope = PyObj(PyDict_New()); // create a scope to execute the scripts in
    auto bootstrap_result = PyObj(PyRun_String(bootstrap.str().c_str(), Py_file_input, scope, scope));
    intPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "int_property"));
    floatPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "float_property"));
    stringPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "string_property"));
    objectPropertyType_ = PyObj::Borrow(PyDict_GetItemString(scope, "object_property"));
    cameraProtocol_ = PyObj::Borrow(PyDict_GetItemString(scope, "Camera"));

    // read the 'devices' field, which must be a dictionary of label->device
    auto deviceDict = PyObj::Borrow(PyDict_GetItemString(scope, "devices"));
    auto metadataDict = PyObj::Borrow(PyDict_GetItemString(scope, "metadata"));

    auto metadata = PyObj(PyDict_Items(metadataDict));
    auto device_count = PyList_Size(metadata);
    for (Py_ssize_t i = 0; i < device_count; i++) {
        auto key_value = PyObj::Borrow(PyList_GetItem(metadata, i));
        auto name = PyObj::Borrow(PyTuple_GetItem(key_value, 0));
        auto data = PyObj::Borrow(PyTuple_GetItem(key_value, 1));
        auto type = PyObj::Borrow(PyTuple_GetItem(data, 0));
        auto device = PyObj::Borrow(PyTuple_GetItem(data, 1));
        devices_[name.as<string>()] = { device, type.as<string>() };
    }
    name_ = scriptPath.stem().generic_string();
    g_hubs[name_] = this;

    return CheckError();
}

PyObj CPyHub::GetDevice(const string& device_id) noexcept {
    auto colon_pos = device_id.find(':');
    if (colon_pos == string::npos)
        return PyObj(); // invalid device string

    auto hub_name = device_id.substr(0, colon_pos);
    auto object_name = device_id.substr(colon_pos + 1);
    auto hub_idx = g_hubs.find(hub_name);
    if (hub_idx == g_hubs.end())
        return PyObj(); // hub not found
    auto hub = hub_idx->second;

    auto device_idx = hub->devices_.find(object_name);
    if (device_idx == hub->devices_.end())
        return PyObj(); // device not found

    return device_idx->second.object;
}





/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CPyCamera::SnapImage()
{
    PyLock lock;
    auto return_value = PyObj(PyObject_CallNoArgs(triggerFunction_));
    return_value = PyObj(PyObject_CallNoArgs(waitFunction_));
    return CheckError();
}

int CPyCamera::Initialize() {
    PyLock lock;
    _check_(PyCameraClass::Initialize());

    triggerFunction_ = object_.Get("trigger");
    waitFunction_ = object_.Get("wait");
    return CheckError();
}

int CPyCamera::Shutdown() {
    StopSequenceAcquisition();
    lastImage_.Clear();
    triggerFunction_.Clear();
    waitFunction_.Clear();
    return PyCameraClass::Shutdown();
}

/**
* Returns pixel data.
* Required by the MM::Camera API.
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
* The calling program allso assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
*/
const unsigned char* CPyCamera::GetImageBuffer()
{
    PyLock lock;
    lastImage_ = object_.Get("image");
    if (CheckError() != DEVICE_OK)
        return nullptr;

    if (!PyArray_Check(lastImage_)) {
        this->LogMessage("Error, 'image' property should return a numpy array");
        return nullptr;
    }
    auto buffer = (PyArrayObject*)lastImage_.get();
    if (PyArray_NDIM(buffer) != 2 || PyArray_TYPE(buffer) != NPY_UINT16 || !(PyArray_FLAGS(buffer) & NPY_ARRAY_C_CONTIGUOUS)) {
        this->LogMessage("Error, 'image' property should be a 2-dimensional numpy array that is c-contiguous in memory and contains 16 bit  unsigned integers");
        return nullptr;
    }

    // check if the array has the correct size
    auto w = GetImageWidth();
    auto h = GetImageHeight();
    auto nw = PyArray_DIM(buffer, 0);
    auto nh = PyArray_DIM(buffer, 1);
    if (nw != w || nh != h) {
        auto msg = "Error, 'image' dimensions should be (" + std::to_string(w) + ", " + std::to_string(h) + ") pixels, but were found to be (" + std::to_string(nw) + ", " + std::to_string(nh) + ") pixels";
        this->LogMessage(msg.c_str());
        return nullptr;
    }

    return (const unsigned char*)PyArray_DATA(buffer);
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageWidth() const
{
    return object_.Get("width").as<long>();
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageHeight() const
{
    return object_.Get("height").as<long>();
}

/**
* Returns image buffer pixel depth in bytes.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageBytesPerPixel() const
{
    return (GetBitDepth() + 7) / 8;
}

/**
* Returns the bit depth (dynamic range) of the pixel. Fixed at 16 bit per pixel
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetBitDepth() const
{
    return 16;
}

/**
* Returns the size in bytes of the image buffer.
* Required by the MM::Camera API.
*/
long CPyCamera::GetImageBufferSize() const
{
    return GetImageWidth() * GetImageHeight() * GetImageBytesPerPixel();
}

/**
* Sets the camera Region Of Interest.
* Required by the MM::Camera API.
* This command will change the dimensions of the image.
* Depending on the hardware capabilities the camera may not be able to configure the
* exact dimensions requested - but should try do as close as possible.
* If the hardware does not have this capability the software should simulate the ROI by
* appropriately cropping each frame.
*/
int CPyCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
    if (xSize == 0 && ySize == 0) // special case: reset ROI
        return ClearROI();

    // apply ROI
    PyLock lock; // make sure all four elements of the ROI are set without any other thread having access in between
    object_.Set("width", (long)xSize);
    object_.Set("height", (long)ySize);
    object_.Set("top", (long)y);
    object_.Set("left", (long)x);
    return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* If multiple ROIs are set, then the returned ROI should encompass all of them.
* Required by the MM::Camera API.
*/
int CPyCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
    PyLock lock; // make sure all four elements of the ROI are read without any other thread having access
    xSize = object_.Get("width").as<long>();
    ySize = object_.Get("height").as<long>();
    x = object_.Get("left").as<long>();
    y = object_.Get("top").as<long>();
    return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int CPyCamera::ClearROI()
{
    PyLock lock; // make sure all four elements of the ROI are set without any other thread having access in between
    double width, height, top, left;
    GetPropertyLowerLimit("top", top);
    GetPropertyLowerLimit("left", left);
    GetPropertyUpperLimit("width", width);
    GetPropertyUpperLimit("height", height);
    object_.Set("width", (long)width);
    object_.Set("height", (long)height);
    object_.Set("top", (long)top);
    object_.Set("left", (long)left);
    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CPyCamera::GetExposure() const
{
    return object_.Get("exposure_ms").as<double>();
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CPyCamera::SetExposure(double exp)
{
    object_.Set("exposure_ms", exp); // cannot directly call SetProperty on python_ because that does not update cached value
    GetCoreCallback()->OnExposureChanged(this, exp);
}

/**
* Returns the current binning factor. Currently only a binning of 1 (no binning) is supported
* Required by the MM::Camera API.
*/
int CPyCamera::GetBinning() const
{
    return 1;
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CPyCamera::SetBinning(int binF)
{
    return binF == 1 ? DEVICE_OK : DEVICE_INVALID_PROPERTY_VALUE;
}

int CPyCamera::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = true;
    return DEVICE_OK;
}
