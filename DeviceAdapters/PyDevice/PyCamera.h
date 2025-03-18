#pragma once
#include "PyDevice.h"
#include "buffer.h"

using PyCameraClass = CPyDeviceTemplate<CLegacyCameraBase<std::monostate>>;
class CPyCamera : public PyCameraClass {
    Py_buffer lastFrame_;
    PyObj read_; // the read() method of the camera object
    
public:
    CPyCamera(const string& id) : PyCameraClass(id)
    {
        lastFrame_.obj = nullptr;
        lastFrame_.buf = nullptr;
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
    void SetExposure(double exp_ms) override;
    int GetBinning() const override;
    int SetBinning(int binF) override;
    int IsExposureSequenceable(bool& isSequenceable) const override;
    int SnapImage() override;
    int Shutdown() override;
    int InsertImage() override;
    int ConnectMethods(const PyObj& methods) override;

private:
    void ReleaseBuffer()
    {
        PyLock lock;
        if (lastFrame_.obj) 
            PyBuffer_Release(&lastFrame_);
        lastFrame_.buf = nullptr;
    }
};