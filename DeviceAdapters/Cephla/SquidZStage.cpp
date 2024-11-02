#include "Squid.h"


const char* g_ZStageName = "ZStage";



SquidZStage::SquidZStage() :
   hub_(0),
   stepSize_um_(0.0),
   screwPitchZmm_(0.3),
   microSteppingDefaultZ_(256),
   fullStepsPerRevZ_(200),
   maxVelocity_(5.0),
   acceleration_(100.0),
   initialized_(false),
   cmdNr_(0)
{
}


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
   int ret = hub_->AssignZStageDevice(this);
   if (ret != DEVICE_OK)
      return ret;

   CPropertyAction* pAct = new CPropertyAction(this, &SquidZStage::OnAcceleration);
   CreateFloatProperty(g_Acceleration, acceleration_, false, pAct);
   SetPropertyLimits(g_Acceleration, 1.0, 6553.5);

   pAct = new CPropertyAction(this, &SquidZStage::OnMaxVelocity);
   CreateFloatProperty(g_Max_Velocity, maxVelocity_, false, pAct);
   SetPropertyLimits(g_Max_Velocity, 1.0, 655.35);

   initialized_ = true;

   return DEVICE_OK;
}


bool SquidZStage::Busy()
{
   return hub_->ZStageBusy();
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
   hub_->GetPositionZSteps(z);
   return DEVICE_OK;
}


int SquidZStage::SetRelativePositionSteps(long zSteps)
{
   return hub_->SendMoveCommand(CMD_MOVE_Z, zSteps);
}


int SquidZStage::Home()
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_HOME_OR_ZERO;
   cmd[2] = AXIS_Z;
   cmd[3] = int((STAGE_MOVEMENT_SIGN_Z + 1) / 2); // "move backward" if SIGN is 1, "move forward" if SIGN is - 1
   int ret = hub_->SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}


int SquidZStage::Stop()
{
   return DEVICE_UNSUPPORTED_COMMAND;
}


int SquidZStage::Callback(long zSteps)
{
   this->GetCoreCallback()->OnStagePositionChanged(this, zSteps * stepSize_um_);
   return DEVICE_OK;
}


int SquidZStage::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(acceleration_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(acceleration_);
      return hub_->SetMaxVelocityAndAcceleration(AXIS_Z, maxVelocity_, acceleration_);
   }
   return DEVICE_OK;
}


int SquidZStage::OnMaxVelocity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxVelocity_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxVelocity_);
      return hub_->SetMaxVelocityAndAcceleration(AXIS_Y, maxVelocity_, acceleration_);
   }
   return DEVICE_OK;
}
