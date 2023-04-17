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
#include "ImgBuffer.h"

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
        this->SetErrorText(ERR_PYTHON_MISSING_PROPERTY, "The Python class is missing a required property, check CoreLog error log for details");

        _python.Construct(this);
    }
    virtual ~CPyDeviceBase() {
    }
    int Initialize() override {
        auto result = _python.Initialize(this);
        if (result != DEVICE_OK) {
            Shutdown();
            return result;
        }
        result = InitializeDevice();
        if (result != DEVICE_OK) {
            Shutdown();
            return result;
        }
        return DEVICE_OK;
    }

    int Shutdown() override {
        return _python.Destruct();
    }
    void GetName(char* name) const override {
        CDeviceUtils::CopyLimitedString(name, _adapterName);
    }
    virtual bool Busy() override {
        return false;
    }
protected:
    /**
    * Called after construction of the Python class to check if all required properties are present and have the correct type,
    * and perform other initialization if needed.
    */
    virtual int InitializeDevice() {
        return DEVICE_OK;
    }
};

class CPyGenericDevice : public CPyDeviceBase<CGenericBase<PythonBridge>> {
    using BaseClass = CPyDeviceBase<CGenericBase<PythonBridge>>;
public:
    constexpr static const char* g_adapterName = "PyDevice";
    CPyGenericDevice() : BaseClass(g_adapterName) {
    }
};

class CPyCamera : public CPyDeviceBase<CCameraBase<PythonBridge>> {
    MM::MMTime readoutStartTime_;
    PyObj lastImage_;       // numpy array corresponding to the last image (prevents deletion during processing)
    PyObj triggerFunction_; // 'trigger' function of the camera object
    PyObj waitFunction_;    // 'wait' function of the camera object
    
    using BaseClass = CPyDeviceBase<CCameraBase<PythonBridge>>;
public:
    constexpr static const char* g_adapterName = "PyCamera";
    CPyCamera() : BaseClass(g_adapterName), readoutStartTime_(0) {
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
protected:
    int InitializeDevice() override;
};
#endif //_Pydevice_H_
