#include "XYStage.h"
#include "UC2Hub.h"
#include <sstream>

XYStage::XYStage() :
   initialized_(false),
   hub_(nullptr),
   posXSteps_(0),
   posYSteps_(0),
   stepSizeUm_(0.05) // example step size (in microns per step)
{
}

XYStage::~XYStage()
{
   Shutdown();
}

int XYStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   hub_ = dynamic_cast<UC2Hub*>(GetParentHub());
   if (!hub_)
      return ERR_NO_PORT_SET;

   initialized_ = true;
   return DEVICE_OK;
}

int XYStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

void XYStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_XYStageName);
}

bool XYStage::Busy()
{
   // Implement hardware busy check if available; otherwise, return false.
   return false;
}

int XYStage::SetPositionSteps(long x, long y)
{
   if (!initialized_) return DEVICE_NOT_CONNECTED;

   // Example JSON command for an absolute move on X (stepperid 1) and Y (stepperid 2)
   std::ostringstream ss;
   ss << R"({"task":"/motor_act","motor":{"steppers":[)"
      << R"({"stepperid":1,"position":)" << x << R"(,"speed":5000,"isabs":1},)"
      << R"({"stepperid":2,"position":)" << y << R"(,"speed":5000,"isabs":1})"
      << "]}}";

   std::string reply;
   int ret = hub_->SendJsonCommand(ss.str(), reply);
   if (ret != DEVICE_OK)
      return ret;

   posXSteps_ = x;
   posYSteps_ = y;
   return DEVICE_OK;
}

int XYStage::GetPositionSteps(long& x, long& y)
{
   if (!initialized_) return DEVICE_NOT_CONNECTED;
   // Return cached positions (or poll hardware if available)
   x = posXSteps_;
   y = posYSteps_;
   return DEVICE_OK;
}

int XYStage::SetRelativePositionSteps(long x, long y)
{
   return SetPositionSteps(posXSteps_ + x, posYSteps_ + y);
}

int XYStage::Home()
{
   // If your hardware supports homing, send the command here.
   // For now, just zero the cached positions.
   posXSteps_ = 0;
   posYSteps_ = 0;
   return DEVICE_OK;
}

int XYStage::Stop()
{
   // If hardware supports a stop command, send it here.
   return DEVICE_OK;
}

int XYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
   // Provide valid limits if known; otherwise, use placeholder values.
   xMin = 0;
   xMax = 100000;
   yMin = 0;
   yMax = 100000;
   return DEVICE_OK;
}

int XYStage::IsXYStageSequenceable(bool& isSequenceable) const
{
   isSequenceable = false;
   return DEVICE_OK;
}

int XYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
   // Convert step limits to microns using the step size
   long lxMin, lxMax, lyMin, lyMax;
   int ret = GetStepLimits(lxMin, lxMax, lyMin, lyMax);
   if (ret != DEVICE_OK)
      return ret;

   xMin = lxMin * stepSizeUm_;
   xMax = lxMax * stepSizeUm_;
   yMin = lyMin * stepSizeUm_;
   yMax = lyMax * stepSizeUm_;
   return DEVICE_OK;
}

int XYStage::SetOrigin()
{
   // Define the current position as the origin (0,0).
   posXSteps_ = 0;
   posYSteps_ = 0;
   return DEVICE_OK;
}
