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
        
        _python.Construct(this);
    }
    virtual ~CPyDeviceBase() {
    }
    int Initialize() override {
        return _python.Initialize(this);
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
};

class CPyGenericDevice : public CPyDeviceBase<CGenericBase<PythonBridge>> {
    using BaseClass = CPyDeviceBase<CGenericBase<PythonBridge>>;
public:
    constexpr static const char* g_adapterName = "Generic Python device";
    CPyGenericDevice() : BaseClass(g_adapterName) {
    }
};

class CPyCamera : public CPyDeviceBase<CCameraBase<PythonBridge>> {
    MM::MMTime readoutStartTime_;
    using BaseClass = CPyDeviceBase<CCameraBase<PythonBridge>>;
public:
    constexpr static const char* g_adapterName = "Generic Python device";
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

    int StartSequenceAcquisition(double interval) override;
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) override;
    int StopSequenceAcquisition() override;
    int IsExposureSequenceable(bool& isSequenceable) const override;

    int SnapImage() override;
};
#endif //_Pydevice_H_
