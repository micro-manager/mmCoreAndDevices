#include "pch.h"
#include "PyCamera.h"

#include "buffer.h"

const char* g_Keyword_Width = "Width";
const char* g_Keyword_Height = "Height";
const char* g_Keyword_Top = "Top";
const char* g_Keyword_Left = "Left";
const char* g_Keyword_Exposure = "Exposure-ms";
const char* g_Keyword_Binning = "Binning";
const char* g_Method_Read = "read";

/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/

int CPyCamera::ConnectMethods(const PyObj& methods)
{
    _check_(PyCameraClass::ConnectMethods(methods));
    read_ = methods.GetDictItem("read");
    return CheckError();
}

int CPyCamera::SnapImage()
{
    auto frame = read_.Call();
    ReleaseBuffer();
    if (PyObject_GetBuffer(frame, &lastFrame_, PyBUF_C_CONTIGUOUS) == -1)
        this->LogMessage("Error, 'image' property should return a numpy array");
    return CheckError();
}

int CPyCamera::Shutdown()
{
    StopSequenceAcquisition();
    ReleaseBuffer();
    return PyCameraClass::Shutdown();
}

/**
* Returns pixel data.
* Required by the MM::Camera API.
* The calling program will assume the size of the buffer based on the values
* obtained from GetImageBufferSize(), which in turn should be consistent with
* values returned by GetImageWidth(), GetImageHeight() and GetImageBytesPerPixel().
* The calling program also assumes that camera never changes the size of
* the pixel buffer on its own. In other words, the buffer can change only if
* appropriate properties are set (such as binning, pixel type, etc.)
* 
*/
const unsigned char* CPyCamera::GetImageBuffer()
{
    PyLock lock;
    if (CheckError() != DEVICE_OK)
        return nullptr;

    if (lastFrame_.buf == nullptr || lastFrame_.ndim != 2 || lastFrame_.itemsize != 2) {
        this->LogMessage(
            "Error, 'image' property should be a 2-dimensional numpy array that is c-contiguous in memory and contains 16 bit  unsigned integers");
        return nullptr;
    }

    // check if the array has the correct size
    auto w = GetImageWidth();
    auto h = GetImageHeight();
    auto nw = lastFrame_.shape[1];
    auto nh = lastFrame_.shape[0];
    if (nw != w || nh != h)
    {
        auto msg = "Error, 'image' dimensions should be (" + std::to_string(w) + ", " + std::to_string(h) +
            ") pixels, but were found to be (" + std::to_string(nw) + ", " + std::to_string(nh) + ") pixels";
        this->LogMessage(msg.c_str());
        return nullptr;
    }

    return static_cast<const unsigned char*>(lastFrame_.buf);
}

/**
* Returns image buffer X-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageWidth() const
{
    return GetLongProperty(g_Keyword_Width);
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageHeight() const
{
    return GetLongProperty(g_Keyword_Height);
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
    SetLongProperty(g_Keyword_Left, x);
    SetLongProperty(g_Keyword_Top, y);
    SetLongProperty(g_Keyword_Width, xSize);
    SetLongProperty(g_Keyword_Height, ySize);
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
    x = GetLongProperty(g_Keyword_Left);
    y = GetLongProperty(g_Keyword_Top);
    xSize = GetLongProperty(g_Keyword_Width);
    ySize = GetLongProperty(g_Keyword_Height);
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
    GetPropertyLowerLimit(g_Keyword_Top, top);
    GetPropertyLowerLimit(g_Keyword_Left, left);
    GetPropertyUpperLimit(g_Keyword_Width, width);
    GetPropertyUpperLimit(g_Keyword_Height, height);
    SetLongProperty(g_Keyword_Top, static_cast<long>(top));
    SetLongProperty(g_Keyword_Left, static_cast<long>(left));
    SetLongProperty(g_Keyword_Width, static_cast<long>(width));
    SetLongProperty(g_Keyword_Height, static_cast<long>(height));
    return DEVICE_OK;
}


/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CPyCamera::GetExposure() const
{
    return GetFloatProperty(g_Keyword_Exposure);
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CPyCamera::SetExposure(double exp_ms)
{
    if (SetFloatProperty(g_Keyword_Exposure, exp_ms) == DEVICE_OK)
        GetCoreCallback()->OnExposureChanged(this, exp_ms);
}

/**
* Returns the current binning factor. Currently only a binning of 1 (no binning) is supported
* Required by the MM::Camera API.
*/
int CPyCamera::GetBinning() const
{
    return GetLongProperty(g_Keyword_Binning);
}

/**
* Sets binning factor.
* Required by the MM::Camera API.
*/
int CPyCamera::SetBinning(int binF)
{
    return SetLongProperty(g_Keyword_Binning, binF);
}

int CPyCamera::IsExposureSequenceable(bool& isSequenceable) const
{
    isSequenceable = false;
    return DEVICE_OK;
}

// overriding default implementation which is broken (does not check for nullptr return from buffer)
int CPyCamera::InsertImage()
{
    char label[MM::MaxStrLength];
    this->GetLabel(label);
    Metadata md;
    md.put(MM::g_Keyword_Metadata_CameraLabel, label);
    auto buffer = GetImageBuffer();
    if (!buffer)
        return DEVICE_ERR;

    return GetCoreCallback()->InsertImage(this, buffer, GetImageWidth(),
                                             GetImageHeight(), GetImageBytesPerPixel(),
                                             md.Serialize().c_str());
}
