#include "Squid.h"


const char* g_ZStageName = "ZStage";



SquidZStage::SquidZStage() :
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
   stepSize_um_ = 0.001 / (screwPitchZmm_ / (microSteppingDefaultZ_ * fullStepsPerRevZ_));

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


 double SquidZStage::GetStepSize() 
 { 
    return stepSize_um_; 
 }


 int SquidZStage::SetPositionSteps(long z)
 {
    return DEVICE_OK;
 }


 int GetPositionSteps(long& z);
 int SetRelativePositionSteps(long z);
 int Home();
 int Stop();

};