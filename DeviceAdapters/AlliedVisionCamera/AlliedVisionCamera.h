/*=============================================================================
  Copyright (C) 2012 - 2023 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

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
#ifndef AlliedVisionCamera_H
#define AlliedVisionCamera_H

#include <array>
#include <functional>
#include <unordered_map>

#include "DeviceBase.h"
#include "Loader/LibLoader.h"
#include "PropertyItem.h"

///////////////////////////////////////////////////////////////////////////////
// STATIC FEATURE NAMES (FROM VIMBA)
///////////////////////////////////////////////////////////////////////////////
static constexpr const char* g_PixelFormatFeature = "PixelFormat";
static constexpr const char* g_ExposureFeature = "ExposureTime";
static constexpr const char* g_BinningHorizontalFeature = "BinningHorizontal";
static constexpr const char* g_BinningVerticalFeature = "BinningVertical";
static constexpr const char* g_Width = "Width";
static constexpr const char* g_Height = "Height";
static constexpr const char* g_OffsetX = "OffsetX";
static constexpr const char* g_OffsetY = "OffsetY";
static constexpr const char* g_WidthMax = "WidthMax";
static constexpr const char* g_HeightMax = "HeightMax";

///////////////////////////////////////////////////////////////////////////////
// STATIC VARIABLES
///////////////////////////////////////////////////////////////////////////////
static constexpr const char* g_True = "True";
static constexpr const char* g_False = "False";
static constexpr const char* g_Execute = "Execute";
static constexpr const char* g_Command = "Command";
static constexpr const char* g_ChunkCategory = "ChunkDataControl";
static constexpr const char* g_EventCategory = "EventControl";
static constexpr const char* g_AcquisitionStart = "AcquisitionStart";
static constexpr const char* g_AcquisitionStop = "AcquisitionStop";
static constexpr const char* g_AcqusitionStatus = "AcqusitionStatus";

/**
 * @brief Global pointer to the Vimba API, that needs to be released in a
 * correct way at the end
 */
static std::unique_ptr<VimbaXApi> g_api{nullptr};

/**
 * @brief Main Allied Vision Camera class
 */
class AlliedVisionCamera : public CCameraBase<AlliedVisionCamera> {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
 public:
  /**
   * @brief Contructor of Allied Vision Camera
   * @param[in] deviceName  Device name
   * @param[in] sdk         Unique pointer to the SDK
   */
  AlliedVisionCamera(const char* deviceName, std::unique_ptr<VimbaXApi>& sdk);
  /**
   * @brief Allied Vision Camera destructor
   */
  virtual ~AlliedVisionCamera();

  ///////////////////////////////////////////////////////////////////////////////
  // uMANAGER API METHODS
  ///////////////////////////////////////////////////////////////////////////////
  int Initialize() override;
  int Shutdown() override;
  const unsigned char* GetImageBuffer() override;
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
  int GetROI(unsigned& x, unsigned& y, unsigned& xSize,
             unsigned& ySize) override;
  int ClearROI() override;
  int IsExposureSequenceable(bool& isSequenceable) const override;
  void GetName(char* name) const override;
  int StartSequenceAcquisition(double interval_ms) override;
  int StartSequenceAcquisition(long numImages, double interval_ms,
                               bool stopOnOverflow) override;
  int StopSequenceAcquisition() override;
  bool IsCapturing() override;

  ///////////////////////////////////////////////////////////////////////////////
  // uMANAGER CALLBACKS
  ///////////////////////////////////////////////////////////////////////////////
  int OnPixelType(MM::PropertyBase* pProp,
                  MM::ActionType eAct);  //!<< PixelType property callback
  int OnBinning(MM::PropertyBase* pProp,
                MM::ActionType eAct);  //!<< Binning property callback
  int onProperty(MM::PropertyBase* pProp,
                 MM::ActionType eAct);  //!<< General property callback

  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
 private:
  // Static variables
  static constexpr const VmbUint8_t MAX_FRAMES =
      7;  //!<< Max frame number in the buffer

  /**
   * @brief Setup error messages for Vimba API
   */
  void setApiErrorMessages();

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
   * @param[in] callback            uManager callback for given property
   * @return VmbError_t
   */
  VmbError_t createPropertyFromFeature(const VmbFeatureInfo_t* feature,
                                       MM::ActionFunctor* callback);

  /**
   * @brief Helper method to set allowed values for given property, based on
   * its feature type
   * @param[in] feature         Vimba feature name
   * @param[in] propertyName    uManager propery name (if differs from
   * feature name)
   * @return
   */
  VmbError_t setAllowedValues(const VmbFeatureInfo_t* feature,
                              const char* propertyName);

  /**
   * @brief Insert ready frame to the uManager
   * @param[in] frame   Pointer to the frame
   */
  void insertFrame(VmbFrame_t* frame);

  /**
   * @brief Method to get feature value, based on its type. Feature value is
   * always a string type.
   * @param[in] featureInfo     Feature info object
   * @param[in] featureName     Feature name
   * @param[out] value          Value of feature, read from device
   * @return VmbError_t
   */
  VmbError_t getFeatureValue(VmbFeatureInfo_t* featureInfo,
                             const char* featureName, std::string& value);
  /**
   * @brief Method to get feature value, based on its type. Feature value is
   * always a string type.
   * @param[in] featureName     Feature name
   * @param[out] value          Value of feature, read from device
   * @return VmbError_t
   */
  VmbError_t getFeatureValue(const char* featureName, std::string& value);

  /**
   * @brief Method to set a feature value, bases on its type. Feature value is
   * always a string type.
   * @param[in] featureInfo     Feature info object
   * @param[in] featureName     Feature name
   * @param[in] value           Value of feature to be set
   * @return VmbError_t
   */
  VmbError_t setFeatureValue(VmbFeatureInfo_t* featureInfo,
                             const char* featureName, std::string& value);

  /**
   * @brief Method to set a feature value, bases on its type. Feature value is
   * always a string type.
   * @param[in] featureName     Feature name
   * @param[in] value           Value of feature to be set
   * @return VmbError_t
   */
  VmbError_t setFeatureValue(const char* featureName, std::string& value);

  /**
   * @brief Helper method to map feature name into property name of uManager
   * @param[in] feature     Vimba Feature name
   * @param property        uManager property name
   */
  void mapFeatureNameToPropertyName(const char* feature,
                                    std::string& property) const;

  /**
   * @brief Helper method to map uManager property in Vimba feature or features
   * name
   * @param[in] property    uManager property name
   * @param featureNames    Vimba feature or features name
   */
  void mapPropertyNameToFeatureNames(
      const char* property, std::vector<std::string>& featureNames) const;

  /**
   * @brief In case trying to set invalid value, adjust it to the closest with
   * inceremntal step
   * @param[in] min     Minimum for given property
   * @param[in] max     Maximum for given property
   * @param[in] step    Incremental step
   * @param[in] propertyValue   Value that was tried to be set
   * @return Adjusted value resresented as a string
   */
  std::string adjustValue(double min, double max, double step,
                          double propertyValue) const;

  ///////////////////////////////////////////////////////////////////////////////
  // MEMBERS
  ///////////////////////////////////////////////////////////////////////////////
  std::unique_ptr<VimbaXApi>& m_sdk;             //<! Unique pointer to the SDK
  VmbHandle_t m_handle;                          //<! Device handle
  std::string m_cameraName;                      //<! Camera name
  std::array<VmbFrame_t, MAX_FRAMES> m_frames;   //<! Frames array
  std::array<VmbUint8_t*, MAX_FRAMES> m_buffer;  //<! Images buffers

  VmbUint32_t m_bufferSize;     //<! Buffer size (the same for every frame)
  bool m_isAcquisitionRunning;  //<! Sequence acquisition status (true if
                                // running)
  std::unordered_map<std::string, PropertyItem>
      m_propertyItems;  //!< Internal map of properties
  static const std::unordered_map<std::string, std::string>
      m_featureToProperty;  //!< Map of features name into uManager properties
  static const std::unordered_multimap<std::string, std::string>
      m_propertyToFeature;  //!< Map of uManager properties into Vimba features
};

#endif
