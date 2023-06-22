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

#include "DeviceBase.h"
#include "Loader/LibLoader.h"

/**
 * @brief Pointer to the Vimba API
*/
static std::unique_ptr<VimbaXApi> g_api;

/**
 * @brief Main Allied Vision Camera class
*/
class AlliedVisionCamera : public CCameraBase<AlliedVisionCamera> {
  // PUBLIC
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

  // API Methods
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

  // Callbacks
  int OnPixelTypeChanged(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnBinningChanged(MM::PropertyBase* pProp, MM::ActionType eAct);

  // Static variables
  static constexpr const VmbUint8_t MAX_FRAMES = 7;

  // PRIVATE
 private:
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
  VmbError_t createPropertyFromFeature(const VmbFeatureInfo_t* feature);

  /**
   * @brief Insert ready frame to the uManager
   * @param[in] frame   Pointer to the frame
  */
  void insertFrame(VmbFrame_t* frame);

  // MEMBERS
  VmbHandle_t m_handle;                         //<! Device handle
  std::string m_cameraName;                     //<! Camera name
  std::array<VmbFrame_t, MAX_FRAMES> m_frames;  //<! Frames array
  std::array<VmbUint8_t*, MAX_FRAMES> m_buffer; //<! Images buffers

  VmbUint32_t m_bufferSize;                     //<! Buffer size (the same for every frame)
  VmbInt64_t m_imageWidth;                      //<! Image width size
  VmbInt64_t m_imageHeight;                     //<! Image heigh size

  bool m_isAcquisitionRunning;                  //<! Sequence acquisition status (true if running)
};

#endif
