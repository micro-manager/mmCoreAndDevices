/*=============================================================================
  Copyright (C) 2023 Allied Vision Technologies.  All Rights Reserved.

  This file is distributed under the BSD license.
  License text is included with the source distribution.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/
#ifndef ALLIEDVISIONCAMERA_H
#define ALLIEDVISIONCAMERA_H

#include <array>
#include <functional>
#include <regex>
#include <unordered_map>
#include <unordered_set>

#include "AlliedVisionDeviceBase.h"
#include "DeviceBase.h"
#include "Loader/LibLoader.h"

///////////////////////////////////////////////////////////////////////////////
// STATIC FEATURE NAMES (FROM VIMBA)
///////////////////////////////////////////////////////////////////////////////
static constexpr const char *g_PixelFormatFeature = "PixelFormat";
static constexpr const char *g_ExposureFeature = "ExposureTime";
static constexpr const char *g_ExposureAbsFeature = "ExposureTimeAbs";
static constexpr const char *g_BinningHorizontalFeature = "BinningHorizontal";
static constexpr const char *g_BinningVerticalFeature = "BinningVertical";
static constexpr const char *g_Width = "Width";
static constexpr const char *g_Height = "Height";
static constexpr const char *g_OffsetX = "OffsetX";
static constexpr const char *g_OffsetY = "OffsetY";
static constexpr const char *g_WidthMax = "WidthMax";
static constexpr const char *g_HeightMax = "HeightMax";

///////////////////////////////////////////////////////////////////////////////
// STATIC VARIABLES
///////////////////////////////////////////////////////////////////////////////
static constexpr const char *g_True = "True";
static constexpr const char *g_False = "False";
static constexpr const char *g_Execute = "Execute";
static constexpr const char *g_Command = "Command";
static constexpr const char *g_ChunkCategory = "ChunkDataControl";
static constexpr const char *g_EventCategory = "EventControl";
static constexpr const char *g_AcquisitionStart = "AcquisitionStart";
static constexpr const char *g_AcquisitionStop = "AcquisitionStop";
static constexpr const char *g_AcqusitionStatus = "AcqusitionStatus";

static constexpr const double MS_TO_US = 1000.0;

/**
 * @brief Pixel Format class that contains VMB Pixel Format info.
 *
 * 8bit GRAY    [no. component = 1]
 * 10bit GRAY    [no. component = 1]
 * 12bit GRAY    [no. component = 1]
 * 14bit GRAY    [no. component = 1]
 * 16bit GRAY   [no. component = 1]
 * 32bit RGB    [no. component = 4]
 */
class PixelFormatConverter
{
    ///////////////////////////////////////////////////////////////////////////////
    // PUBLIC
    ///////////////////////////////////////////////////////////////////////////////
public:
    /**
     * @brief Constructor
     */
    PixelFormatConverter() :
        m_pixelType{ "Mono8" },
        m_components{ 1 },
        m_bitDepth{ 8 },
        m_bytesPerPixel{ 1 },
        m_isMono{ true },
        m_vmbFormat{ VmbPixelFormatMono8 }
    {
    }

    /**
     * @brief Destructor
     */
    virtual ~PixelFormatConverter() = default;

    /**
     * @brief Setter for pixel type
     * @param[in] type    New pixel type (string value from VMB)
     */
    void setPixelType(const std::string &type)
    {
        m_pixelType = type;
        updateFields();
    }

    /**
     * @brief Getter to check if given pixel type is Mono or Color
     * @return True if Mono, otherwise false
     */
    bool isMono() const
    {
        return m_isMono;
    }

    /**
     * @brief Getter for number of components for given pixel type
     * uManager supports only 1 or 4 components
     * @return Number of components
     */
    unsigned getNumberOfComponents() const
    {
        return m_components;
    }

    /**
     * @brief Getter for bit depth for given pixel type
     * @return Number of bits for one pixel
     */
    unsigned getBitDepth() const
    {
        return m_bitDepth;
    }

    /**
     * @brief Getter for bytes per pixel
     * @return Bytes per pixel
     */
    unsigned getBytesPerPixel() const
    {
        return m_bytesPerPixel;
    }

    /**
     * @brief Getter of destination VmbPixelFormat.
     * 1. Mono8
     * 2. Mono10
     * 3. Mono12
     * 4. Mono14
     * 5. Mono16
     * 6. RGB32
     *
     * These types fits into following VmbPixelTypes:
     * 1. VmbPixelFormatMono8
     * 2. VmbPixelFormatMono10
     * 3. VmbPixelFormatMono12
     * 4. VmbPixelFormatMono14
     * 5. VmbPixelFormatMono16
     * 6. VmbPixelFormatBgra8
     * @return Destination VmbPixelFormat
     */
    VmbPixelFormatType getVmbFormat() const
    {
        return m_vmbFormat;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // PRIVATE
    ///////////////////////////////////////////////////////////////////////////////
private:
    /**
     * @brief Helper method to update info if given format is Mono or Color
     */
    void updateMono()
    {
        std::regex re("Mono(\\d+)");
        std::smatch m;
        m_isMono = std::regex_search(m_pixelType, m, re);
    }

    /**
     * @brief Helper method to update number of components for given pixel type
     * uManager supports only 1 or 4 components
     */
    void updateNumberOfComponents()
    {
        m_components = m_isMono ? 1 : 4;
    }

    /**
     * @brief Helper method to update bit depth for given pixel type
     */
    void updateBitDepth()
    {
        m_bitDepth = 8;
        if (isMono())
        {
            std::regex re("Mono(\\d+)");
            std::smatch m;
            std::regex_search(m_pixelType, m, re);
            if (m.size() > 0)
            {
                if (std::atoi(m[1].str().c_str()) == 16) // Mono16
                {
                    m_vmbFormat = VmbPixelFormatMono16;
                    m_bitDepth = 16;
                }
                else if (std::atoi(m[1].str().c_str()) == 14) // Mono14
                {
                    m_vmbFormat = VmbPixelFormatMono14;
                    m_bitDepth = 16;
                }
                else if (std::atoi(m[1].str().c_str()) == 12) // Mono12
                {
                    m_vmbFormat = VmbPixelFormatMono12;
                    m_bitDepth = 16;
                }
                else if (std::atoi(m[1].str().c_str()) == 10) // Mono10
                {
                    m_vmbFormat = VmbPixelFormatMono10;
                    m_bitDepth = 16;
                }
                else // Default to Mono8
                {
                    m_vmbFormat = VmbPixelFormatMono8;
                    m_bitDepth = 8;
                }
            }
            else
            {
                // ERROR
            }
        }
        else
        {
            m_vmbFormat = VmbPixelFormatBgra8;
            m_bitDepth = 32;
        }
    }

    /**
     * @brief Helper method to update bytes per pixel
     */
    void updateBytesPerPixel()
    {
        m_bytesPerPixel = m_bitDepth / 8;
    }

    /**
     * @brief Helper method to update all required fields
     */
    void updateFields()
    {
        // [IMPORTANT] Keep order of the calls
        updateMono();
        updateNumberOfComponents();
        updateBitDepth();
        updateBytesPerPixel();
    }

    std::string m_pixelType;        //!< Pixel type (in string) - value from VMB
    unsigned m_components;          //!< Number of components
    unsigned m_bitDepth;            //<! Bit depth
    unsigned m_bytesPerPixel;       //!< Bytes per pixel
    bool m_isMono;                  //!< Mono or Color
    VmbPixelFormatType m_vmbFormat; //!< Destination VmbPixelFormatType
};

/**
 * @brief Main Allied Vision Camera class
 */
class AlliedVisionCamera : public AlliedVisionDeviceBase<CLegacyCameraBase<AlliedVisionCamera>, AlliedVisionCamera>
{
    ///////////////////////////////////////////////////////////////////////////////
    // PUBLIC
    ///////////////////////////////////////////////////////////////////////////////
public:
    /**
     * @brief Contructor of Allied Vision Camera
     * @param[in] deviceName  Device name
     */
    AlliedVisionCamera(const char *deviceName);
    /**
     * @brief Allied Vision Camera destructor
     */
    virtual ~AlliedVisionCamera();

    ///////////////////////////////////////////////////////////////////////////////
    // uMANAGER API METHODS
    ///////////////////////////////////////////////////////////////////////////////
    int Initialize() override;
    int Shutdown() override;
    const unsigned char *GetImageBuffer() override;
    unsigned GetImageWidth() const override;
    unsigned GetImageHeight() const override;
    unsigned GetImageBytesPerPixel() const override;
    int SnapImage() override;
    long GetImageBufferSize() const override;
    unsigned GetBitDepth() const override;
    int GetBinning() const override;
    int SetBinning(int binSize) override;
    void SetExposure(double exp_ms) override;
    double GetExposure() const override;
    int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) override;
    int GetROI(unsigned &x, unsigned &y, unsigned &xSize, unsigned &ySize) override;
    int ClearROI() override;
    int IsExposureSequenceable(bool &isSequenceable) const override;
    void GetName(char *name) const override;
    int StartSequenceAcquisition(double interval_ms) override;
    int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) override;
    int StopSequenceAcquisition() override;
    bool IsCapturing() override;
    unsigned GetNumberOfComponents() const override;

    ///////////////////////////////////////////////////////////////////////////////
    // uMANAGER CALLBACKS
    ///////////////////////////////////////////////////////////////////////////////
    int onProperty(MM::PropertyBase *pProp,
                   MM::ActionType eAct); //!<< General property callback

    ///////////////////////////////////////////////////////////////////////////////
    // PRIVATE
    ///////////////////////////////////////////////////////////////////////////////
private:
    // Static variables
    static constexpr const VmbUint8_t MAX_FRAMES = 7; //!<< Max frame number in the buffer

    /**
     * @brief Helper method to handle change of pixel type
     * @param[in] pixelType   New pixel type (as string)
     */
    void handlePixelFormatChange(const std::string &pixelType);

    /**
     * @brief Resize all buffers for image frames
     * @return VmbError_t
     */
    VmbError_t resizeImageBuffer();

    /**
     * @brief Setup uManager properties from Vimba features
     * @return VmbError_t
     */
    VmbError_t setupProperties();

    /**
     * @brief Helper method to create single uManager property from Vimba feature
     * @param[in] feature             Pointer to the Vimba feature
     * @return VmbError_t
     */
    VmbError_t createPropertyFromFeature(const VmbFeatureInfo_t *feature);

    /**
     * @brief Helper method to set allowed values for given property, based on
     * its feature type
     * @param[in] feature         Vimba feature name
     * @param[in] propertyName    uManager propery name (if differs from
     * feature name)
     * @return
     */
    VmbError_t setAllowedValues(const VmbFeatureInfo_t *feature, const char *propertyName);

    /**
     * @brief Insert ready frame to the uManager
     * @param[in] frame   Pointer to the frame
     */
    void insertFrame(VmbFrame_t *frame);

    /**
     * @brief Method to get feature value, based on its type. Feature value is
     * always a string type.
     * @param[in] featureInfo     Feature info object
     * @param[in] featureName     Feature name
     * @param[out] value          Value of feature, read from device
     * @return VmbError_t
     */
    VmbError_t getFeatureValue(VmbFeatureInfo_t *featureInfo, const char *featureName, std::string &value) const;
    /**
     * @brief Method to get feature value, based on its type. Feature value is
     * always a string type.
     * @param[in] featureName     Feature name
     * @param[out] value          Value of feature, read from device
     * @return VmbError_t
     */
    VmbError_t getFeatureValue(const char *featureName, std::string &value) const;

    /**
     * @brief Method to set a feature value, bases on its type. Feature value is
     * always a string type.
     * @param[in] featureInfo     Feature info object
     * @param[in] featureName     Feature name
     * @param[in] value           Value of feature to be set
     * @return VmbError_t
     */
    VmbError_t setFeatureValue(VmbFeatureInfo_t *featureInfo, const char *featureName, std::string &value);

    /**
     * @brief Method to set a feature value, bases on its type. Feature value is
     * always a string type.
     * @param[in] featureName     Feature name
     * @param[in] value           Value of feature to be set
     * @return VmbError_t
     */
    VmbError_t setFeatureValue(const char *featureName, std::string &value);

    /**
     * @brief Helper method to map feature name into property name of uManager
     * @param[in] feature     Vimba Feature name
     * @return                uManager property name
     */
    std::string mapFeatureNameToPropertyName(const char *feature) const;

    /**
     * @brief Helper method to map uManager property in Vimba feature or features
     * name
     * @param[in] property    uManager property name
     * @param featureNames    Vimba feature or features name
     */
    std::string mapPropertyNameToFeatureName(const char *property) const;

    /**
     * @brief In case trying to set invalid value, adjust it to the closest with
     * inceremntal step

     * @param[in] step    Incremental step

     * @return Adjusted value resresented as a string
     */

    /**
     * @brief In case trying to set invalid value, adjust it to the closest with
     * inceremntal step
     * @param[in] featureInfo     Feature info object
     * @param[in] min             Minimum for given property
     * @param[in] max             Maximum for given property
     * @param[in] propertyValue   Value that was tried to be set
     * @return Adjusted value resresented as a string
     */
    std::string adjustValue(VmbFeatureInfo_t &featureInfo, double min, double max, double propertyValue) const;

    /**
     * @brief Internal method to transform image to the destination format that
     * uManager supports (see \ref PixelFormatConverter) and replaces input frame
     * with output (transformed) frame
     * @param[in] frame   Frame with image to transform from and into
     * @return Error in case of failure
     */
    VmbError_t transformImage(VmbFrame_t *frame);

    ///////////////////////////////////////////////////////////////////////////////
    // MEMBERS
    ///////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<VimbaXApi> m_sdk;              //<! Shared pointer to the SDK
    VmbHandle_t m_handle;                          //<! Device handle
    std::string m_cameraName;                      //<! Camera name
    std::array<VmbFrame_t, MAX_FRAMES> m_frames;   //<! Frames array for uManager
    std::array<VmbUint8_t *, MAX_FRAMES> m_buffer; //<! Images buffers for uManager

    VmbUint32_t m_bufferSize;    //<! Buffer size of image for uManager
    VmbUint32_t m_payloadSize;   //<! Payload size of image for Vimba
    bool m_isAcquisitionRunning; //<! Sequence acquisition status (true if running)

    std::string m_exposureFeatureName;

    PixelFormatConverter m_currentPixelFormat;                                     //<! Current Pixel Format information
    static const std::unordered_map<std::string, std::string> m_featureToProperty; //!< Map of features name into uManager properties
    std::unordered_map<std::string, std::string> m_propertyToFeature;              //!< Map of uManager properties into Vimba features
    
    static const std::unordered_set<std::string> m_ipAddressFeatures;
    static const std::unordered_set<std::string> m_macAddressFeatures;
};

#endif
