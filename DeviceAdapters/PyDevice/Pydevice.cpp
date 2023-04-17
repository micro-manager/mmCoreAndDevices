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

#include "PyDevice.h"
#include <numpy/arrayobject.h>



/**
* Performs exposure and grabs a single image.
* This function should block during the actual exposure and return immediately afterwards
* (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
* Required by the MM::Camera API.
*/
int CPyCamera::SnapImage()
{
    readoutStartTime_ = GetCurrentMMTime();
    _python.CallMethod(triggerFunction_);
    _python.CallMethod(waitFunction_);
    return DEVICE_OK;
}

int CPyCamera::InitializeDevice() {
    const auto required_properties = { "width", "height", "top", "left", "exposure_ms", "image", "trigger", "wait"};
    bool missing = false;
    for (auto p : required_properties) {
        try {
            _python.GetProperty(p);
        }
        catch (PyObj::NullPointerException) {
            _python.PythonError();
            missing = true;
        }
    }
    if (missing)
        return ERR_PYTHON_MISSING_PROPERTY;
    
    triggerFunction_ = _python.GetProperty("trigger");
    waitFunction_ = _python.GetProperty("wait");
    return DEVICE_OK;
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
    if (!PyArray_API)
        import_array(); // initialize numpy again!

    lastImage_ = _python.GetProperty("image");
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
* todo: cache width and height properties?
*/
unsigned CPyCamera::GetImageWidth() const
{
    long width = 0;
    _python.GetProperty("width", width);
    return width;
}

/**
* Returns image buffer Y-size in pixels.
* Required by the MM::Camera API.
*/
unsigned CPyCamera::GetImageHeight() const
{
    long height = 0;
    _python.GetProperty("height", height);
    return height;
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
    _python.SetProperty("width", (long)xSize);
    _python.SetProperty("height", (long)ySize);
    _python.SetProperty("top", (long)y);
    _python.SetProperty("left", (long)x);
    return DEVICE_OK;
}

/**
* Returns the actual dimensions of the current ROI.
* If multiple ROIs are set, then the returned ROI should encompass all of them.
* Required by the MM::Camera API.
*/
int CPyCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
    long txSize, tySize, tx, ty;
    _python.GetProperty("width", txSize);
    _python.GetProperty("height", tySize);
    _python.GetProperty("top", ty);
    _python.GetProperty("left", tx);
    x = tx; // not needed if we can be sure that unsigned has same size as long, but this is not guaranteed by c++
    y = ty;
    xSize = txSize;
    ySize = tySize;
    return DEVICE_OK;
}

/**
* Resets the Region of Interest to full frame.
* Required by the MM::Camera API.
*/
int CPyCamera::ClearROI()
{
    double width, height, top, left;
    GetPropertyLowerLimit("top", top);
    GetPropertyLowerLimit("left", left);
    GetPropertyUpperLimit("width", width);
    GetPropertyUpperLimit("height", height);
    _python.SetProperty("width", (long)width);
    _python.SetProperty("height", (long)height);
    _python.SetProperty("top", (long)top);
    _python.SetProperty("left", (long)left);
    return DEVICE_OK;
}

/**
* Returns the current exposure setting in milliseconds.
* Required by the MM::Camera API.
*/
double CPyCamera::GetExposure() const
{
    double exposure = 0;
    _python.GetProperty("exposure_ms", exposure); // cannot use GetProperty of CDeviceBase because that is not const !?
    return exposure;
}

/**
* Sets exposure in milliseconds.
* Required by the MM::Camera API.
*/
void CPyCamera::SetExposure(double exp)
{
    SetProperty("exposure_ms", std::to_string(exp).c_str()); // cannot directly call SetProperty on _python because that does not update cached value
    GetCoreCallback()->OnExposureChanged(this, exp);;
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
    isSequenceable = false;
    return DEVICE_OK;
}

/**
 * Required by the MM::Camera API
 * Please implement this yourself and do not rely on the base class implementation
 * The Base class implementation is deprecated and will be removed shortly
 */
int CPyCamera::StartSequenceAcquisition(double interval)
{
    return DEVICE_NOT_SUPPORTED; // sequence mode is not supported yet
}
int CPyCamera::StopSequenceAcquisition()
{
    return DEVICE_NOT_SUPPORTED; // sequence mode is not supported yet
}
int CPyCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
    return DEVICE_NOT_SUPPORTED; // sequence mode is not supported yet
}
///////////////////////////////////////////////////////////////////////////////
// Private CPyCamera methods
///////////////////////////////////////////////////////////////////////////////

/*
int CPyCamera::GeneratePythonImage(ImgBuffer& img)
{

    MMThreadGuard g(imgPixelsLock_);

    //std::string pixelType;
    char buf[MM::MaxStrLength];
    GetProperty(MM::g_Keyword_PixelType, buf);
    std::string pixelType(buf);

    if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
        return 0;

    unsigned imgWidth = img.Width();

    // Define all neccesary for bridge:
    auto modname = PyUnicode_FromString("Pyscanner");
    auto funcname = "cpp_single_capture";

    // define all the input variable seperately


    auto input = PyUnicode_FromString(("['" + dacportin_ + "']").c_str());
    auto output = PyUnicode_FromString(("['" + dacportoutx_ + "', '" + dacportouty_ + "']").c_str());

    // read the image settings
    auto height = std::to_string(ScanXSteps_);
    auto width = std::to_string(ScanYSteps_);

    auto resolution = PyUnicode_FromString(("[" + height + ", " + width + "]").c_str());

    std::string zoomFactorString = std::to_string(zoomFactor_);
    auto zoom = PyUnicode_FromString(("[" + zoomFactorString + "]").c_str());


    std::string delayString = std::to_string(delay_);
    auto delay = PyUnicode_FromString(("[" + delayString + "]").c_str());




    auto dwelltimefactor = dwelltime_ / 1000000;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(12) << dwelltimefactor;
    auto dwelltime = PyUnicode_FromString(("[" + stream.str() + "]").c_str());

    std::string scanpaddingString = std::to_string(scanpadding_);
    auto scanpadding = PyUnicode_FromString(("[" + scanpaddingString + "]").c_str());


    std::string inputminString = std::to_string(inputmin_);
    std::string inputmaxString = std::to_string(inputmax_);

    auto input_range = PyUnicode_FromString(("[" + inputminString + ", " + inputmaxString + "]").c_str());

    auto module = PyImport_Import(modname);

    auto func = PyObject_GetAttrString(module, funcname);
    auto args = PyTuple_New(8);

    PyTuple_SetItem(args, 0, input);
    PyTuple_SetItem(args, 1, output);
    PyTuple_SetItem(args, 2, resolution);
    PyTuple_SetItem(args, 3, zoom);
    PyTuple_SetItem(args, 4, delay);
    PyTuple_SetItem(args, 5, dwelltime);
    PyTuple_SetItem(args, 6, scanpadding);
    PyTuple_SetItem(args, 7, input_range);

    auto returnvalue = PyObject_CallObject(func, args);
    Py_DECREF(args);

    import_array();

    PyObject* numpy = PyImport_ImportModule("numpy");

    // Convert the returnvalue to a numpy array
    PyObject* array = PyArray_FROM_OTF(returnvalue, NPY_DOUBLE, NPY_IN_ARRAY);

    npy_intp* dimensions = PyArray_DIMS(array);

    // Get a pointer to the data of the numpy array
    double* data = (double*)PyArray_DATA(array);
    long maxValue = (1L << bitDepth_) - 1;
    unsigned j, k;
    if (pixelType.compare(g_PixelType_8bit) == 0)
    {
        unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
        for (int i = 0; i < img.Height() * img.Width() * img.Depth(); i++)
        {
            pBuf[i] = (unsigned char)(data[i] * maxValue);
        }
        Py_DECREF(array);
        Py_DECREF(func);
        Py_DECREF(module);
    }
    else if (pixelType.compare(g_PixelType_16bit) == 0) // this is what we do with 16 bit images
    {
        unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());

        double min_val = inputmin_;
        double max_val = inputmax_;


        // Scale each input value to the range of 16-bit unsigned integers
        for (int i = 0; i < imgWidth * img.Height(); i++) {
            double scaled_val = (data[i] - min_val) / (max_val - min_val) * 65535.0;
            pBuf[i] = (unsigned short)scaled_val;
        }
        //        for (j = 0; j < img.Height(); j++)
        //        {
        //            for (k = 0; k < imgWidth; k++)
        //            {
        //                long lIndex = imgWidth * j + k;
        //                double val = data[imgWidth * j + k];
        //                val = val * maxValue;
        //                *(pBuf + lIndex) = (unsigned short)val;
        //            }
        //        }
                // Decrement the reference count of the numpy array
        Py_DECREF(array);
        Py_DECREF(func);
        Py_DECREF(module);
    }


    return 0;
}*/