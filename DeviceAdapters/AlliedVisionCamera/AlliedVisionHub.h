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
#ifndef ALLIEDVISIONHUB_H
#define ALLIEDVISIONHUB_H

#include "AlliedVisionDeviceBase.h"
#include "DeviceBase.h"
#include "Loader/LibLoader.h"

#include <memory>
///////////////////////////////////////////////////////////////////////////////
// STATIC VARIABLES
///////////////////////////////////////////////////////////////////////////////
static constexpr const char *g_hubName = "AlliedVisionHub";

/**
 * @brief Class that represents a HUB of supported devices
 */
class AlliedVisionHub
    : public AlliedVisionDeviceBase<HubBase<AlliedVisionHub>, AlliedVisionHub> {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
public:
  /**
   * @brief Contructor of a HUB
   */
  AlliedVisionHub();

  /**
   * @brief Destructor of a HUB
   */
  virtual ~AlliedVisionHub() = default;

  /**
   * @brief SDK getter
   * @return Pointer to SDK
   */
  std::shared_ptr<VimbaXApi> &getSDK();

  ///////////////////////////////////////////////////////////////////////////////
  // uMANAGER API METHODS
  ///////////////////////////////////////////////////////////////////////////////
  int DetectInstalledDevices() override;
  int Initialize() override;
  int Shutdown() override;
  void GetName(char *name) const override;
  bool Busy() override;

  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
private:
  std::shared_ptr<VimbaXApi> m_sdk; //<! Shared pointer to the SDK
};

#endif
