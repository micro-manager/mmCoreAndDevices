#include "Squid.h"


const char* g_ZStageName = "ZStage";



SquidZStage::SquidZStage() :
   hub_(0),
   stepSize_um_(0.0),
   screwPitchZmm_(0.3),
   microSteppingDefaultZ_(256),
   fullStepsPerRevZ_(200),
   initialized_(false)
{}


SquidZStage::~SquidZStage()
{
   if (initialized_)
      Shutdown();
}


int SquidZStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}


void SquidZStage::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_ZStageName);
}


int SquidZStage::Initialize()
{
   stepSize_um_ = -1000.0 * screwPitchZmm_ / (microSteppingDefaultZ_ * fullStepsPerRevZ_); 

   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   return DEVICE_OK;
}


// TODO: implement
bool SquidZStage::Busy()
{
   return false;
}


int SquidZStage::SetPositionUm(double pos)
{
   return SetPositionSteps((long)(pos / stepSize_um_));
}


int SquidZStage::SetRelativePositionUm(double d)
{
   return SetRelativePositionSteps((long)(d / stepSize_um_));
}


int SquidZStage::GetPositionUm(double& pos) {
   long z;
   int ret = GetPositionSteps(z);
   if (ret != DEVICE_OK)
      return ret;
   pos = z * stepSize_um_;
   return DEVICE_OK;
}



int SquidZStage::SetPositionSteps(long zSteps)
{
   return hub_->SendMoveCommand(CMD_MOVETO_Z, zSteps);
}


int SquidZStage::GetPositionSteps(long& z)
{
   return hub_->GetPositionZSteps(z);
}


int SquidZStage::SetRelativePositionSteps(long zSteps)
{
   return hub_->SendMoveCommand(CMD_MOVE_Z, zSteps);
}


int SquidZStage::Home()
{
   return DEVICE_UNSUPPORTED_COMMAND;
}


int SquidZStage::Stop()
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

