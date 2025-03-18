/**
* 
* Useful confugation settings from 
https://github.com/hongquanli/octopi-research/blob/master/software/configurations/configuration_octopi_v2.ini
stage_movement_sign_x = -1
stage_movement_sign_y = 1
stage_movement_sign_z = -1
stage_movement_sign_theta = 1
stage_pos_sign_x = -1
stage_pos_sign_y = 1
stage_pos_sign_z = -1
stage_pos_sign_theta = 1
tracking_movement_sign_x = 1
tracking_movement_sign_y = 1
tracking_movement_sign_z = 1
tracking_movement_sign_theta = 1
use_encoder_x = False
_use_encoder_x_options=[True,False]
use_encoder_y = False
_use_encoder_y_options=[True,False]
use_encoder_z = False
_use_encoder_z_options=[True,False]
use_encoder_theta = False
_use_encoder_theta_options=[True,False]
encoder_pos_sign_x = 1
encoder_pos_sign_y = 1
encoder_pos_sign_z = 1
encoder_pos_sign_theta = 1
encoder_step_size_x_mm = 100e-6
encoder_step_size_y_mm = 100e-6
encoder_step_size_z_mm = 100e-6
encoder_step_size_theta = 1
fullsteps_per_rev_x = 200
fullsteps_per_rev_y = 200
fullsteps_per_rev_z = 200
fullsteps_per_rev_theta = 200
screw_pitch_x_mm = 2.54
screw_pitch_y_mm = 2.54
screw_pitch_z_mm = 0.3
microstepping_default_x = 256
microstepping_default_y = 256
microstepping_default_z = 256
microstepping_default_theta = 256
x_motor_rms_current_ma = 1000
y_motor_rms_current_ma = 1000
z_motor_rms_current_ma = 500
x_motor_i_hold = 0.25
y_motor_i_hold = 0.25
z_motor_i_hold = 0.5
max_velocity_x_mm = 25
max_velocity_y_mm = 25
max_velocity_z_mm = 5
max_acceleration_x_mm = 500
max_acceleration_y_mm = 500
max_acceleration_z_mm = 100
scan_stabilization_time_ms_x = 160
scan_stabilization_time_ms_y = 160
scan_stabilization_time_ms_z = 20
homing_enabled_x = True
_homing_enabled_x_options=[True,False]
homing_enabled_y = True
_homing_enabled_y_options=[True,False]
homing_enabled_z = True
_homing_enabled_z_options=[True,False]
sleep_time_s = 0.005
*/

# include "Squid.h"


const char* g_XYStageName = "XYStage";

SquidXYStage::SquidXYStage() :
   hub_(0),
   stepSizeX_um_(0.0),
   stepSizeY_um_(0.0),
   fullStepsPerRevX_(200),
   fullStepsPerRevY_(200),
   screwPitchXmm_(2.54),
   screwPitchYmm_(2.54),
   microSteppingDefaultX_(256),
   microSteppingDefaultY_(256),
   posX_um_(0.0),
   posY_um_(0.0),
   busy_(false),
   maxVelocity_(25.0),
   acceleration_(500.0),
   initialized_(false),
   cmdNr_(0)
{
   InitializeDefaultErrorMessages();
}

SquidXYStage::~SquidXYStage()
{
   if (initialized_)
   {
      Shutdown();
   }
}

int SquidXYStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

void SquidXYStage::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_XYStageName);
}

int SquidXYStage::Initialize()
{
   if (initialized_)
   {
      return DEVICE_ERR;
   }

   // minus sign is there to enforce compatibility with MM sense of direction
   stepSizeX_um_ = -1000.0 * screwPitchXmm_ / (microSteppingDefaultX_ * fullStepsPerRevX_); 
   stepSizeY_um_ = -1000.0 * screwPitchYmm_ / (microSteppingDefaultY_ * fullStepsPerRevY_);
    
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   int ret = hub_->AssignXYStageDevice(this);
   if (ret != DEVICE_OK)
      return ret;
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   CPropertyAction* pAct = new CPropertyAction(this, &SquidXYStage::OnAcceleration);
   CreateFloatProperty(g_Acceleration, acceleration_, false, pAct);
   SetPropertyLimits(g_Acceleration, 1.0, 6553.5);

   pAct = new CPropertyAction(this, &SquidXYStage::OnMaxVelocity);
   CreateFloatProperty(g_Max_Velocity, maxVelocity_, false, pAct);
   SetPropertyLimits(g_Max_Velocity, 1.0, 655.35);

   initialized_ = true;

   return DEVICE_OK;
}


bool SquidXYStage::Busy()
{
   return hub_->XYStageBusy();
}


int SquidXYStage::Home()
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_HOME_OR_ZERO;
   cmd[2] = AXIS_XY;
   cmd[3] = int((STAGE_MOVEMENT_SIGN_X + 1) / 2); // "move backward" if SIGN is 1, "move forward" if SIGN is - 1
   cmd[4] = int((STAGE_MOVEMENT_SIGN_Y + 1) / 2); // "move backward" if SIGN is 1, "move forward" if SIGN is - 1
   int ret = hub_->SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}


/*
* Sets the position of the stage in steps
* I believe these are the microsteps of the device
*/
int SquidXYStage::SetPositionSteps(long xSteps, long ySteps)
{
   int ret = hub_->SendMoveCommand(CMD_MOVETO_X, xSteps);
   if (ret != DEVICE_OK)
      return ret;
   return hub_->SendMoveCommand(CMD_MOVETO_Y, ySteps);
}


/*
* Sets the position of the stage in steps
* I believe these are the microsteps of the device
*/
int SquidXYStage::SetRelativePositionSteps(long xSteps, long ySteps)
{
   int ret = hub_->SendMoveCommand(CMD_MOVE_X, xSteps);
   if (ret != DEVICE_OK)
      return ret;
   return hub_->SendMoveCommand(CMD_MOVE_Y, ySteps);
}


int SquidXYStage::Callback(long xSteps, long ySteps)
{
   this->GetCoreCallback()->OnXYStagePositionChanged(this,
      xSteps * stepSizeX_um_, ySteps * stepSizeY_um_);
   return DEVICE_OK;
}


int SquidXYStage::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(acceleration_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(acceleration_);
      int ret = hub_->SetMaxVelocityAndAcceleration(AXIS_X, maxVelocity_, acceleration_);
      if (ret != DEVICE_OK)
         return ret;
      return hub_->SetMaxVelocityAndAcceleration(AXIS_Y, maxVelocity_, acceleration_);
   }
   return DEVICE_OK;
}


int SquidXYStage::OnMaxVelocity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxVelocity_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxVelocity_);
      int ret = hub_->SetMaxVelocityAndAcceleration(AXIS_X, maxVelocity_, acceleration_);
      if (ret != DEVICE_OK)
         return ret;
      return hub_->SetMaxVelocityAndAcceleration(AXIS_Y, maxVelocity_, acceleration_);
   }
   return DEVICE_OK;
}
