#include "PureFocus.h"


const char* g_PureFocusOffsetDeviceName = "PureFocusOffset";
const char* g_PureFocusOffsetDescription = "PureFocusOffset Drive";

PureFocusOffset::PureFocusOffset() :
   initialized_(false),
   pHub_(0),
   stepSize_(0.001)
{
}


PureFocusOffset::~PureFocusOffset()
{
   if (initialized_)
      Shutdown();
}


bool PureFocusOffset::Busy()
{
   if (pHub_ == 0)
      return false;

   return pHub_->IsOffsetLensBusy();
}

void PureFocusOffset::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_PureFocusOffsetDeviceName);
}

int PureFocusOffset::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   pHub_ = static_cast<PureFocusHub*>(GetParentHub());
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   pHub_->SetOffsetDevice(this);

   return DEVICE_OK;
}

int PureFocusOffset::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

// Stage API

int PureFocusOffset::SetPositionUm(double pos)
{
  return SetPositionSteps((long) (pos / stepSize_));
}

int PureFocusOffset::SetPositionSteps(long steps)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   return pHub_->SetOffset(steps);
}

int PureFocusOffset::GetPositionUm(double& pos)
{
   long posSteps;
   int ret = GetPositionSteps(posSteps);
   if (ret != DEVICE_OK)
      return ret;

   pos = (double) (posSteps * stepSize_);
   return DEVICE_OK;
}

int PureFocusOffset::GetPositionSteps(long& steps)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   return pHub_->GetOffset(steps);
}


void PureFocusOffset::CallbackPositionSteps(long steps)
{
   GetCoreCallback()->OnStagePositionChanged(this, steps * stepSize_);
}
