#include "ZStage.h"
#include "UC2Hub.h"
#include <sstream>

ZStage::ZStage() :
   initialized_(false),
   hub_(nullptr),
   posZSteps_(0),
   stepSizeUm_(0.1) // example: 0.1 µm per step
{
}

ZStage::~ZStage()
{
   Shutdown();
}

int ZStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   hub_ = dynamic_cast<UC2Hub*>(GetParentHub());
   if (!hub_)
      return ERR_NO_PORT_SET;

   initialized_ = true;
   return DEVICE_OK;
}

int ZStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

void ZStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_ZStageName);
}

bool ZStage::Busy()
{
   return false;
}

int ZStage::SetPositionSteps(long steps)
{
   if (!initialized_) return DEVICE_NOT_CONNECTED;

   // Example JSON command for an absolute Z move (assume stepperid 3)
   std::ostringstream ss;
   ss << R"({"task":"/motor_act","motor":{"steppers":[)"
      << R"({"stepperid":3,"position":)" << steps << R"(,"speed":2000,"isabs":1})"
      << "]}}";

   std::string reply;
   int ret = hub_->SendJsonCommand(ss.str(), reply);
   if (ret != DEVICE_OK)
      return ret;

   posZSteps_ = steps;
   return DEVICE_OK;
}

int ZStage::GetPositionSteps(long& steps)
{
   if (!initialized_) return DEVICE_NOT_CONNECTED;
   steps = posZSteps_;
   return DEVICE_OK;
}

// Additional helper; not declared in base so remove 'override'
int ZStage::SetRelativePositionSteps(long steps)
{
   return SetPositionSteps(posZSteps_ + steps);
}

int ZStage::Home()
{
   // If the hardware supports homing, send the command; otherwise, zero the cached position.
   posZSteps_ = 0;
   return DEVICE_OK;
}

int ZStage::Stop()
{
   return DEVICE_OK;
}

int ZStage::GetLimits(double& min, double& max)
{
   min = 0.0;
   max = 25000.0; // example: 25,000 µm (25 mm)
   return DEVICE_OK;
}

int ZStage::SetPositionUm(double z)
{
   long steps = static_cast<long>(z / stepSizeUm_);
   return SetPositionSteps(steps);
}

int ZStage::GetPositionUm(double& z)
{
   long steps = 0;
   int ret = GetPositionSteps(steps);
   if (ret != DEVICE_OK)
      return ret;

   z = steps * stepSizeUm_;
   return DEVICE_OK;
}

int ZStage::SetOrigin()
{
   posZSteps_ = 0;
   return DEVICE_OK;
}

int ZStage::IsStageSequenceable(bool& isSequenceable) const
{
   isSequenceable = false;
   return DEVICE_OK;
}

bool ZStage::IsContinuousFocusDrive() const
{
   return false;
}
