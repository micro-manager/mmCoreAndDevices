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
#ifndef ALLIEDVISIONCAMERA_H
#define ALLIEDVISIONCAMERA_H

#include <array>
#include <functional>
#include <unordered_map>

#include "DeviceBase.h"
#include "Loader/LibLoader.h"

/**
 * @brief Pointer to the Vimba API
 */
static std::unique_ptr<VimbaXApi> g_api;

///////////////////////////////////////////////////////////////////////////////
// STATIC FEATURE NAMES (FROM VIMBA)
///////////////////////////////////////////////////////////////////////////////
static constexpr const char* g_PixelFormatFeature = "PixelFormat";
static constexpr const char* g_ExposureFeature = "ExposureTime";
static constexpr const char* g_BinningHorizontalFeature = "BinningHorizontal";
static constexpr const char* g_BinningVerticalFeature = "BinningVertical";
static constexpr const char* g_Width = "Width";
static constexpr const char* g_Height = "Height";

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
   * @param[in] deviceName Device name
   */
  AlliedVisionCamera(const char* deviceName);
  /**
   * @brief Allied Vision Camera destructor
   */
  ~AlliedVisionCamera();

  /**
   * @brief Get connected camera list
   * @return VmbError_t
   */
  static VmbError_t getCamerasList();

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
  int OnExposure(MM::PropertyBase* pProp,
                 MM::ActionType eAct);  //!<< Exposure property callback
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
   * @param[in] feature Pointer to the Vimba feature
   * @return VmbError_t
   */
  /**
   * @brief Helper method to create single uManager property from Vimba feature
   * @param[in] feature             Pointer to the Vimba feature
   * @param[in] callback            uManager callback for given property
   * @param[in] propertyName        uManager propery name (if differs from
   * feature name). By default nullptr
   * @param[in] skipVmbCallback     If set to true, VmbCallback will not be
   * added to this feature. By default false
   * @return VmbError_t
   */
  VmbError_t createPropertyFromFeature(const VmbFeatureInfo_t* feature,
                                       MM::ActionFunctor* callback,
                                       const char* propertyName = nullptr,
                                       bool skipVmbCallback = false);

  /**
   * @brief Helper method to create core properties from feature.
   * @return VmbError_t
   *
   * It is used to create properties which names does not match to the Vimba
   * feature. As example these are Binning, Exposure, PixelType
   */
  VmbError_t createCoreProperties();

  /**
   * @brief Helper method to set allowed values for given property, based on
   * its feature type
   * @param[in] feature         Vimba feature name
   * @param[in] propertyName    uManager propery name (if differs from
   * feature name). By default nullptr
   * @return
   */
  VmbError_t setAllowedValues(const VmbFeatureInfo_t* feature,
                              const char* propertyName = nullptr);

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
   * @brief Method to set a feature value, bases on its type. Feature value is
   * always a string type.
   * @param[in] featureInfo     Feature info object
   * @param[in] featureName     Feature name
   * @param[in] value           Value of feature to be set
   * @return VmbError_t
   */
  VmbError_t setFeatureValue(VmbFeatureInfo_t* featureInfo,
                             const char* featureName, std::string& value);

  ///////////////////////////////////////////////////////////////////////////////
  // MEMBERS
  ///////////////////////////////////////////////////////////////////////////////
  VmbHandle_t m_handle;                          //<! Device handle
  std::string m_cameraName;                      //<! Camera name
  std::array<VmbFrame_t, MAX_FRAMES> m_frames;   //<! Frames array
  std::array<VmbUint8_t*, MAX_FRAMES> m_buffer;  //<! Images buffers

  VmbUint32_t m_bufferSize;     //<! Buffer size (the same for every frame)
  bool m_isAcquisitionRunning;  //<! Sequence acquisition status (true if
                                // running)
};

#endif
