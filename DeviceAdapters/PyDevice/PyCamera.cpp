#include "pch.h"
#include "PyCamera.h"
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>




/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CPyCamera::SnapImage()
{
    object_.CallMember("trigger");
    return CheckError();
}

int CPyCamera::Shutdown() {
    StopSequenceAcquisition();
    lastImage_.Clear();
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
    lastImage_ = object_.CallMember("read");
    if (CheckError() != DEVICE_OK)
        return nullptr;

    if (!PyArray_Check(lastImage_)) {
        this->LogMessage("Error, 'image' property should return a numpy array");
        return nullptr;
    }
    auto buffer = (PyArrayObject*)(PyObject*)lastImage_;
    if (PyArray_NDIM(buffer) != 2 || PyArray_TYPE(buffer) != NPY_UINT16 || !(PyArray_FLAGS(buffer) & NPY_ARRAY_C_CONTIGUOUS)) {
        this->LogMessage("Error, 'image' property should be a 2-dimensional numpy array that is c-contiguous in memory and contains 16 bit  unsigned integers");
        return nullptr;
    }

    // check if the array has the correct size
    auto w = GetImageWidth();
    auto h = GetImageHeight();
    auto nh = PyArray_DIM(buffer, 0);
    auto nw = PyArray_DIM(buffer, 1);
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
    double value_ms;
    const_cast<CPyCamera*>(this)->GetProperty("MeasurementTime", value_ms);
    return value_ms;
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CPyCamera::SetExposure(double value_ms)
{
    if (SetProperty("MeasurementTime", std::to_string(value_ms).c_str()) == DEVICE_OK)
        GetCoreCallback()->OnExposureChanged(this, value_ms);
}

/**
* Returns the current binning factor. Currently only a binning of 1 (no binning) is supported
* Required by the MM::Camera API.
*/
int CPyCamera::GetBinning() const
{
    return object_.Get("binning").as<long>();
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CPyCamera::SetBinning(int binF)
{
    object_.Set("binning", (long)binF);
    return CheckError();
}

int CPyCamera::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = true;
    return DEVICE_OK;
}

// overriding default implementation which is broken (does not check for nullptr return from buffer)
int CPyCamera::InsertImage()
{
    char label[MM::MaxStrLength];
    this->GetLabel(label);
    Metadata md;
    md.put("Camera", label);
    auto buffer = GetImageBuffer();
    if (!buffer)
        return DEVICE_ERR;

    int ret = GetCoreCallback()->InsertImage(this, buffer, GetImageWidth(),
        GetImageHeight(), GetImageBytesPerPixel(),
        md.Serialize().c_str());
    if (!isStopOnOverflow() && ret == DEVICE_BUFFER_OVERFLOW)
    {
        // do not stop on overflow - just reset the buffer
        GetCoreCallback()->ClearImageBuffer(this);
        return GetCoreCallback()->InsertImage(this, buffer, GetImageWidth(),
            GetImageHeight(), GetImageBytesPerPixel(),
            md.Serialize().c_str());
    }
    else
        return ret;
}