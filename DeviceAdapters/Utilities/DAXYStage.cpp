///////////////////////////////////////////////////////////////////////////////
// FILE:          DAXYStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                DAXYStage by Ed Simmon, 11/28/2011
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
//                2015-2016, Open Imaging, Inc.
//                Altos Labs, 2022
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//

#ifdef _WIN32
// Prevent windows.h from defining min and max macros,
// which clash with std::min and std::max.
#define NOMINMAX
#endif

#include "Utilities.h"

extern const char* g_DeviceNameDAXYStage;


DAXYStage::DAXYStage() :
   DADeviceNameX_(""),
   DADeviceNameY_(""),
   initialized_(false),
   minDAVoltX_(0.0),
   maxDAVoltX_(10.0),
   minDAVoltY_(0.0),
   maxDAVoltY_(10.0),
   minStageVoltX_(0.0),
   maxStageVoltX_(5.0),
   minStageVoltY_(0.0),
   maxStageVoltY_(5.0),
   minStagePosX_(0.0),
   maxStagePosX_(200.0),
   minStagePosY_(0.0),
   maxStagePosY_(200.0),
   posX_(0.0),
   posY_(0.0),
   originPosX_(0.0),
   originPosY_(0.0),
   stepSizeXUm_(1),
   stepSizeYUm_(1)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid DA device");
   SetErrorText(ERR_NO_DA_DEVICE, "No DA Device selected");
   SetErrorText(ERR_VOLT_OUT_OF_RANGE, "The DA Device cannot set the requested voltage");
   SetErrorText(ERR_POS_OUT_OF_RANGE, "The requested position is out of range");
   SetErrorText(ERR_NO_DA_DEVICE_FOUND, "No DA Device loaded");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameDAXYStage, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "XYStage controlled with voltage provided by two Digital to Analog outputs", MM::String, true);

}

DAXYStage::~DAXYStage()
{
}

void DAXYStage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameDAXYStage);
}

int DAXYStage::Initialize()
{
   // get list with available DA devices.  
   // TODO: this is a initialization parameter, which makes it harder for the end-user to set up!
   char deviceName[MM::MaxStrLength];
   availableDAs_.clear();
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableDAs_.push_back(std::string(deviceName));
      }
      else
         break;
   }



   CPropertyAction* pAct = new CPropertyAction(this, &DAXYStage::OnDADeviceX);
   std::string defaultXDA = "Undefined";
   if (availableDAs_.size() >= 1)
      defaultXDA = availableDAs_[0];
   CreateProperty("X DA Device", defaultXDA.c_str(), MM::String, false, pAct, false);
   if (availableDAs_.size() >= 1)
      SetAllowedValues("X DA Device", availableDAs_);
   else
      return ERR_NO_DA_DEVICE_FOUND;

   pAct = new CPropertyAction(this, &DAXYStage::OnDADeviceY);
   std::string defaultYDA = "Undefined";
   if (availableDAs_.size() >= 2)
      defaultYDA = availableDAs_[1];
   CreateProperty("Y DA Device", defaultYDA.c_str(), MM::String, false, pAct, false);
   if (availableDAs_.size() >= 1)
      SetAllowedValues("Y DA Device", availableDAs_);
   else
      return ERR_NO_DA_DEVICE_FOUND;


   // This is needed, otherwise DeviceDA_ is not always set resulting in crashes 
   // This could lead to strange problems if multiple DA devices are loaded
   SetProperty("X DA Device", defaultXDA.c_str());
   SetProperty("Y DA Device", defaultYDA.c_str());

   std::ostringstream tmp;
   tmp << DADeviceNameX_;
   LogMessage(tmp.str().c_str());

   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   if (da_x != 0)
      da_x->GetLimits(minDAVoltX_, maxDAVoltX_);

   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_y != 0)
      da_y->GetLimits(minDAVoltY_, maxDAVoltY_);

   // Min volts
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMinVoltX);
   CreateProperty("Stage X Low Voltage", "0", MM::Float, false, pAct, false);
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMinVoltY);
   CreateProperty("Stage Y Low Voltage", "0", MM::Float, false, pAct, false);

   // Max volts
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMaxVoltX);
   CreateProperty("Stage X High Voltage", "5", MM::Float, false, pAct, false);
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMaxVoltY);
   CreateProperty("Stage Y High Voltage", "5", MM::Float, false, pAct, false);

   // Min pos
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMinPosX);
   CreateProperty("Stage X Minimum Position", "0", MM::Float, false, pAct, false);
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMinPosY);
   CreateProperty("Stage Y Minimum Position", "0", MM::Float, false, pAct, false);

   // Max pos
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMaxPosX);
   CreateProperty("Stage X Maximum Position", "200", MM::Float, false, pAct, false);
   pAct = new CPropertyAction(this, &DAXYStage::OnStageMaxPosY);
   CreateProperty("Stage Y Maximum Position", "200", MM::Float, false, pAct, false);

   if (minStageVoltX_ < minDAVoltX_)
      return ERR_VOLT_OUT_OF_RANGE;

   if (minStageVoltY_ < minDAVoltY_)
      return ERR_VOLT_OUT_OF_RANGE;

   originPosX_ = minStagePosX_;
   originPosY_ = minStagePosY_;

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

int DAXYStage::Shutdown()
{
   if (initialized_)
      initialized_ = false;

   return DEVICE_OK;
}

bool DAXYStage::Busy()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if ((da_x != 0) && (da_y != 0))
      return da_x->Busy() || da_y->Busy();

   // If we are here, there is a problem.  No way to report it.
   return false;
}

int DAXYStage::Home()
{
   return DEVICE_OK;

};


int DAXYStage::Stop()
{
   return DEVICE_OK;

};

/*
 * Sets a voltage (in mV) on the DA, relative to the minimum Stage position
 * The origin is NOT taken into account
 */
int DAXYStage::SetPositionSteps(long stepsX, long stepsY)
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   // Interpret steps to be mV
   double voltX = minStageVoltX_ + (stepsX / 1000.0);
   if (voltX >= minStageVoltX_ && voltX <= maxStageVoltX_)
      da_x->SetSignal(voltX);
   else
      return ERR_VOLT_OUT_OF_RANGE;

   double voltY = minStageVoltY_ + (stepsY / 1000.0);
   if (voltY <= maxStageVoltY_ && voltY >= minStageVoltY_)
      da_y->SetSignal(voltY);
   else
      return ERR_VOLT_OUT_OF_RANGE;

   posX_ = voltX / (maxStageVoltX_ - minStageVoltX_) * (maxStagePosX_ - minStagePosX_) + originPosX_;
   posY_ = voltY / (maxStageVoltY_ - minStageVoltY_) * (maxStagePosY_ - minStagePosY_) + originPosY_;

   return DEVICE_OK;
}

int DAXYStage::GetPositionSteps(long& stepsX, long& stepsY)
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   double voltX = 0, voltY = 0;
   int ret = da_x->GetSignal(voltX);
   if (ret != DEVICE_OK) {
      stepsX = (long)((posX_ + originPosX_) / (maxStagePosX_ - minStagePosX_) * (maxStageVoltX_ - minStageVoltX_) * 1000.0);
   }
   else {
      stepsX = (long)((voltX - minStageVoltX_) * 1000.0);
   }
   ret = da_y->GetSignal(voltY);
   if (ret != DEVICE_OK) {
      stepsY = (long)((posY_ + originPosY_) / (maxStagePosY_ - minStagePosY_) * (maxStageVoltY_ - minStageVoltY_) * 1000.0);
   }
   else {
      stepsY = (long)((voltY - minStageVoltY_) * 1000.0);
   }
   return DEVICE_OK;
}

int DAXYStage::SetRelativePositionSteps(long x, long y)
{
   long xSteps, ySteps;
   GetPositionSteps(xSteps, ySteps);

   return this->SetPositionSteps(xSteps + x, ySteps + y);
}

int DAXYStage::SetPositionUm(double x, double y)
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   double voltX = ((x - originPosX_) / (maxStagePosX_ - minStagePosX_)) * (maxStageVoltX_ - minStageVoltX_);
   if (voltX > maxStageVoltX_ || voltX < minStageVoltX_)
      return ERR_POS_OUT_OF_RANGE;
   double voltY = ((y - originPosY_) / (maxStagePosY_ - minStagePosY_)) * (maxStageVoltY_ - minStageVoltY_);
   if (voltY > maxStageVoltY_ || voltY < minStageVoltY_)
      return ERR_POS_OUT_OF_RANGE;

   //posY_ = y;


   int ret = da_x->SetSignal(voltX);
   if (ret != DEVICE_OK) return ret;
   ret = da_y->SetSignal(voltY);
   if (ret != DEVICE_OK) return ret;

   return ret;
}

int DAXYStage::GetPositionUm(double& x, double& y)
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   double voltX, voltY;
   int ret = da_x->GetSignal(voltX);
   if (ret != DEVICE_OK)
      // DA Device cannot read, set position from cache
      x = posX_;
   else
      x = voltX / (maxStageVoltX_ - minStageVoltX_) * (maxStagePosX_ - minStagePosX_) + originPosX_;

   ret = da_y->GetSignal(voltY);
   if (ret != DEVICE_OK)
      // DA Device cannot read, set position from cache
      y = posY_;
   else
      y = voltY / (maxStageVoltY_ - minStageVoltY_) * (maxStagePosY_ - minStagePosY_) + originPosY_;

   return DEVICE_OK;
}


/*
 * Sets the origin (relative position 0) to the current absolute position
 */
int DAXYStage::SetOrigin()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());

   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   double voltX, voltY;
   int ret = da_x->GetSignal(voltX);
   if (ret != DEVICE_OK)
      return ret;
   ret = da_y->GetSignal(voltY);
   if (ret != DEVICE_OK)
      return ret;

   // calculate absolute current position:
   originPosX_ = voltX / (maxStageVoltX_ - minStageVoltX_) * (maxStagePosX_ - minStagePosX_);

   if (originPosX_ < minStagePosX_ || originPosX_ > maxStagePosX_)
      return ERR_POS_OUT_OF_RANGE;

   return DEVICE_OK;
}

int DAXYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
   xMin = minStagePosX_;
   xMax = maxStagePosX_;
   yMin = minStagePosY_;
   yMax = maxStagePosY_;
   return DEVICE_OK;
}

int DAXYStage::GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int DAXYStage::IsXYStageSequenceable(bool& isSequenceable) const
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   bool x, y;
   da_x->IsDASequenceable(x);
   da_y->IsDASequenceable(y);
   isSequenceable = x && y;
   return DEVICE_OK;
}

int DAXYStage::GetXYStageSequenceMaxLength(long& nrEvents) const
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

long x, y;
int ret = da_x->GetDASequenceMaxLength(x);
if (ret != DEVICE_OK) return ret;
ret = da_y->GetDASequenceMaxLength(y);
if (ret != DEVICE_OK) return ret;
nrEvents = (std::min)(x, y);
return ret;
}

int DAXYStage::StartXYStageSequence()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   int ret = da_x->StartDASequence();
   if (ret != DEVICE_OK) return ret;
   ret = da_y->StartDASequence();
   if (ret != DEVICE_OK) return ret;
   return ret;
}

int DAXYStage::StopXYStageSequence()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   int ret = da_x->StopDASequence();
   if (ret != DEVICE_OK) return ret;
   ret = da_y->StopDASequence();
   if (ret != DEVICE_OK) return ret;
   return DEVICE_OK;
}

int DAXYStage::ClearXYStageSequence()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   int ret = da_x->ClearDASequence();
   if (ret != DEVICE_OK) return ret;
   ret = da_y->ClearDASequence();
   if (ret != DEVICE_OK) return ret;
   return DEVICE_OK;
}

int DAXYStage::AddToXYStageSequence(double positionX, double positionY)
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   double voltageX, voltageY;

   voltageX = ((positionX + originPosX_) / (maxStagePosX_ - minStagePosX_)) *
      (maxStageVoltX_ - minStageVoltX_);
   if (voltageX > maxStageVoltX_)
      voltageX = maxStageVoltX_;
   else if (voltageX < minStageVoltX_)
      voltageX = minStageVoltX_;

   voltageY = ((positionY + originPosY_) / (maxStagePosY_ - minStagePosY_)) *
      (maxStageVoltY_ - minStageVoltY_);
   if (voltageY > maxStageVoltY_)
      voltageY = maxStageVoltY_;
   else if (voltageY < minStageVoltY_)
      voltageY = minStageVoltY_;

   int ret = da_x->AddToDASequence(voltageX);
   if (ret != DEVICE_OK) return ret;

   ret = da_y->AddToDASequence(voltageY);
   if (ret != DEVICE_OK) return ret;

   return DEVICE_OK;
}

int DAXYStage::SendXYStageSequence()
{
   MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceNameX_.c_str());
   MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceNameY_.c_str());
   if (da_x == 0 || da_y == 0)
      return ERR_NO_DA_DEVICE;

   int ret = da_x->SendDASequence();
   if (ret != DEVICE_OK) return ret;
   ret = da_y->SendDASequence();
   return ret;
}

void DAXYStage::UpdateStepSize()
{
   stepSizeXUm_ = (maxStagePosX_ - minStagePosX_) / (maxStageVoltX_ - minStageVoltX_) / 1000.0;
   stepSizeYUm_ = (maxStagePosY_ - minStagePosY_) / (maxStageVoltY_ - minStageVoltY_) / 1000.0;

   std::ostringstream tmp;
   tmp << "Updated stepsize of DA XY stage to x: " << stepSizeXUm_ << " y: " << stepSizeYUm_;
   LogMessage(tmp.str().c_str());
}


///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int DAXYStage::OnDADeviceX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DADeviceNameX_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string DADeviceName;
      pProp->Get(DADeviceName);
      MM::SignalIO* da_x = (MM::SignalIO*)GetDevice(DADeviceName.c_str());
      if (da_x != 0) {
         DADeviceNameX_ = DADeviceName;
      }
      else
         return ERR_INVALID_DEVICE_NAME;
      if (initialized_)
      {
         da_x->GetLimits(minDAVoltX_, maxDAVoltX_);
         UpdateStepSize();
      }
   }
   return DEVICE_OK;
}

int DAXYStage::OnDADeviceY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DADeviceNameY_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string DADeviceName;
      pProp->Get(DADeviceName);
      MM::SignalIO* da_y = (MM::SignalIO*)GetDevice(DADeviceName.c_str());
      if (da_y != 0) {
         DADeviceNameY_ = DADeviceName;
      }
      else
         return ERR_INVALID_DEVICE_NAME;
      if (initialized_)
      {
         da_y->GetLimits(minDAVoltY_, maxDAVoltY_);
         UpdateStepSize();
      }
   }
   return DEVICE_OK;
}



int DAXYStage::OnStageMinVoltX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStageVoltX_);
   }
   else if (eAct == MM::AfterSet)
   {
      double minStageVolt;
      pProp->Get(minStageVolt);
      if (minStageVolt >= minDAVoltX_ && minStageVolt < maxDAVoltX_)
         minStageVoltX_ = minStageVolt;
      else
         return ERR_VOLT_OUT_OF_RANGE;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMaxVoltX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStageVoltX_);
   }
   else if (eAct == MM::AfterSet)
   {
      double maxStageVolt;
      pProp->Get(maxStageVolt);
      if (maxStageVolt > minDAVoltX_ && maxStageVolt <= maxDAVoltX_)
         maxStageVoltX_ = maxStageVolt;
      else
         return ERR_VOLT_OUT_OF_RANGE;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMinVoltY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStageVoltY_);
   }
   else if (eAct == MM::AfterSet)
   {
      double minStageVolt;
      pProp->Get(minStageVolt);
      if (minStageVolt >= minDAVoltY_ && minStageVolt < maxDAVoltY_)
         minStageVoltY_ = minStageVolt;
      else
         return ERR_VOLT_OUT_OF_RANGE;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMaxVoltY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStageVoltY_);
   }
   else if (eAct == MM::AfterSet)
   {
      double maxStageVolt;
      pProp->Get(maxStageVolt);
      if (maxStageVolt > minDAVoltY_ && maxStageVolt <= maxDAVoltY_)
         maxStageVoltY_ = maxStageVolt;
      else
         return ERR_VOLT_OUT_OF_RANGE;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMinPosX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStagePosX_);
   }
   else if (eAct == MM::AfterSet)
   {
      double minStagePos;
      pProp->Get(minStagePos);
      minStagePosX_ = minStagePos;
      UpdateStepSize();
   }

   return DEVICE_OK;
}

int DAXYStage::OnStageMinPosY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStagePosY_);
   }
   else if (eAct == MM::AfterSet)
   {
      double minStagePos;
      pProp->Get(minStagePos);
      minStagePosY_ = minStagePos;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMaxPosX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStagePosX_);
   }
   else if (eAct == MM::AfterSet)
   {
      double maxStagePos;
      pProp->Get(maxStagePos);
      maxStagePosX_ = maxStagePos;
      UpdateStepSize();
   }
   return DEVICE_OK;
}

int DAXYStage::OnStageMaxPosY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStagePosY_);
   }
   else if (eAct == MM::AfterSet)
   {
      double maxStagePos;
      pProp->Get(maxStagePos);
      maxStagePosY_ = maxStagePos;
      UpdateStepSize();
   }
   return DEVICE_OK;
}
