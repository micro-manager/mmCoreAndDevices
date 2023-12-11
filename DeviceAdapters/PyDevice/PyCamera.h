#pragma once
#include "PyDevice.h"

using PyCameraClass = CPyDeviceTemplate<CCameraBase<std::monostate>>;
class CPyCamera : public PyCameraClass {
    PyObj lastFrame_; // numpy array corresponding to the last image, we hold a reference count so that we are sure the array does not get deleted during processing */

public:
    CPyCamera(const string& id) : PyCameraClass(id) {}
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
    int InsertImage() override;
};