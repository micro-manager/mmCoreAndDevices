#include "PureFocus.h"

const char* g_PureFocusAutoFocusDeviceName = "PureFocusAutoFocus";
const char* g_PureFocusAutoFocusDescription = "PureFocusAutoFocus Device";
const char* g_SampleDetected = "Sample Detected";
const char* g_InFocus = "In Focus";

PureFocusAutoFocus::PureFocusAutoFocus() :
   initialized_(false),
   pHub_(0),
   pinholeCenter_(0),
   pinholeWidth_(0),
   laserPower_(0),
   objective_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_NOT_IN_RANGE, "Sample is not in range");
}


PureFocusAutoFocus::~PureFocusAutoFocus()
{
   if (initialized_)
      Shutdown();
}

int PureFocusAutoFocus::Shutdown()
{
   if (initialized_)
   {
      if (pHub_ != 0)
         pHub_->SetAutofocusDevice(0);

      initialized_ = false;
      pHub_ = 0;
   }

   return DEVICE_OK;
}


void PureFocusAutoFocus::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_PureFocusAutoFocusDeviceName);
}

int PureFocusAutoFocus::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   pHub_ = static_cast<PureFocusHub*>(GetParentHub());
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   // Focus score
   CPropertyAction* pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnFocusScore);
   int ret = CreateProperty("Focus Score", "0", MM::Integer, true, pAct);
   // Device API
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnFocus);
   ret = CreateProperty(g_InFocus, "0", MM::Integer, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnSampleDetected);
   ret = CreateProperty(g_SampleDetected, "0", MM::Integer, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Pinhole Columns
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnPinholeCenter);
   ret = CreateProperty("Pinhole Columns", "1", MM::Integer, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Pinhole Width
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnPinholeWidth);
   ret = CreateProperty("Pinhole Width", "1", MM::Integer, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Laser Power
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnLaserPower);
   ret = CreateProperty("Laser Power", "2048", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Set allowable ranges
   SetPropertyLimits("Laser Power", 0, 4095);

   // Objective
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnObjective);
   ret = CreateProperty("Objective", "1", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits("Objective", 1, 6);

   // List properties of selected objective
   // Would only need to update when objective changes
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnList);
   ret = CreateProperty("List", "", MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Servo on/off
   pAct = new CPropertyAction(this, &PureFocusAutoFocus::OnServo);
   ret = CreateProperty("Servo", "Off", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   AddAllowedValue("Servo", "On");
   AddAllowedValue("Servo", "Off");

   pHub_->SetAutofocusDevice(this);

   initialized_ = true;

   return DEVICE_OK;

}

bool PureFocusAutoFocus::Busy()
{
   // TODO: need to figure out how to best implement this
   return false;
}


int PureFocusAutoFocus::SetContinuousFocusing(bool state)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (state)
   {
      if (!pHub_->IsSampleDetected())
      {
         return ERR_NOT_IN_RANGE;
      }
   }
   return pHub_->SetServo(state);
}

int PureFocusAutoFocus::GetContinuousFocusing(bool& state)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   return pHub_->GetServo(state);
}

bool PureFocusAutoFocus::IsContinuousFocusLocked()
{
   if (pHub_ == 0)
      return false;

   return pHub_->IsInFocus();
}

int PureFocusAutoFocus::FullFocus()
{
   if (pHub_ == 0)
      return false;

   int ret = SetContinuousFocusing(true);
   if (ret != DEVICE_OK)
      return ret;

   // TODO: wait for a timeout or until we are in focus

   return SetContinuousFocusing(false);
}

int PureFocusAutoFocus::IncrementalFocus()
{
   return FullFocus();
}

int PureFocusAutoFocus::GetCurrentFocusScore(double& score)
{
   if (pHub_ == 0)
      return false;

   return pHub_->GetFocusScore(score);
}

int PureFocusAutoFocus::GetLastFocusScore(double& score)
{
   return GetCurrentFocusScore(score);
}

int PureFocusAutoFocus::GetOffset(double& offset)
{
   if (pHub_ == 0)
      return false;

   long lOffset;
   int ret = pHub_->GetOffset(lOffset);
   if (ret != DEVICE_OK)
      return ret;

   // We do not try to convert this number to anything
   offset = (double)lOffset;
   return DEVICE_OK;
}

int PureFocusAutoFocus::SetOffset(double offset)
{
   if (pHub_ == 0)
      return false;

   return pHub_->SetOffset((long) offset);
}

int PureFocusAutoFocus::OnServo(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      // Query actual device status
      bool state;
      int ret = pHub_->GetServo(state);
      if (ret != DEVICE_OK)
         return ret;

      if (state)
         pProp->Set("On");
      else
         pProp->Set("Off");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string state;
      pProp->Get(state);

      bool lockState = (state == "On");
      int ret = pHub_->SetServo(lockState);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnFocusScore(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      double score;
      int ret = pHub_->GetFocusScore(score);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(score);
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

    if (eAct == MM::BeforeGet)
    {
        std::string focus = pHub_->IsInFocus() ? "1" : "0";
        pProp->Set(focus.c_str());
    }
    return DEVICE_OK;
}


int PureFocusAutoFocus::OnSampleDetected(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

    if (eAct == MM::BeforeGet)
    {
        std::string sampleDetected = pHub_->IsSampleDetected() ? "1" : "0";
        pProp->Set(sampleDetected.c_str());
    }
    return DEVICE_OK;
}

int PureFocusAutoFocus::OnPinholeCenter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      int ret = pHub_->GetPinholeProperties(pinholeCenter_, pinholeWidth_);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)pinholeCenter_);
   }
   else if (eAct == MM::AfterSet)
   {
      long center;
      pProp->Get(center);

      int ret = pHub_->SetPinholeProperties(center, pinholeWidth_);
      if (ret != DEVICE_OK)
         return ret;

      pinholeCenter_ = (int)center;
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnPinholeWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      int ret = pHub_->GetPinholeProperties(pinholeCenter_, pinholeWidth_);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)pinholeWidth_);
   }
   else if (eAct == MM::AfterSet)
   {
      long width;
      pProp->Get(width);

      int ret = pHub_->SetPinholeProperties(pinholeCenter_, width);
      if (ret != DEVICE_OK)
         return ret;

      pinholeWidth_ = (int) width;
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      int ret = pHub_->GetLaserPower(laserPower_);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)laserPower_);
   }
   else if (eAct == MM::AfterSet)
   {
      long power;
      pProp->Get(power);

      // Validate the range
      if (power < 0 || power > 4095)
         return DEVICE_INVALID_PROPERTY_VALUE;

      return pHub_->SetLaserPower(laserPower_);
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnObjective(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      int ret = pHub_->GetObjective(objective_);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)objective_);
   }
   else if (eAct == MM::AfterSet)
   {
      long objective;
      pProp->Get(objective);

      // Validate the range
      if (objective < 1 || objective > 6)
         return DEVICE_INVALID_PROPERTY_VALUE;

      int ret = pHub_->SetObjective(objective);
      if (ret != DEVICE_OK)
         return ret;

      objective_ = (int)objective;
   }

   return DEVICE_OK;
}

int PureFocusAutoFocus::OnList(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (pHub_ == 0)
      return ERR_DEVICE_NOT_FOUND;

   if (eAct == MM::BeforeGet)
   {
      std::string list;
      int ret = pHub_->GetList(list);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(list.c_str());
   }
   return DEVICE_OK;
}

void PureFocusAutoFocus::CallbackSampleDetected(bool detected)
{
   GetCoreCallback()->OnPropertyChanged(this, g_SampleDetected, detected ? "1" : "0");
}

void PureFocusAutoFocus::CallbackInFocus(bool inFocus)
{
   GetCoreCallback()->OnPropertyChanged(this, g_InFocus, inFocus ? "1" : "0");
}
