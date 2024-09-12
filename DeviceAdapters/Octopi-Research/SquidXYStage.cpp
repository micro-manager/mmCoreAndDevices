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
   fullStepsPerRevX_(200),
   fullStepsPerRevY_(200),
   screwPitchXmm_(2.54),
   screwPitchYmm_(2.54),
   microSteppingDefaultX_(256),
   microSteppingDefaultY_(256),
   posX_um_(0.0),
   posY_um_(0.0),
   busy_(false),
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



   stepSizeX_um_ = 0.001 / (screwPitchXmm_ / (microSteppingDefaultX_ * fullStepsPerRevX_));
   stepSizeY_um_ = 0.001 / (screwPitchYmm_ / (microSteppingDefaultY_ * fullStepsPerRevY_));

   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   initialized_ = true;

   return DEVICE_OK;
}

bool SquidXYStage::Busy() 
{
   // TODO!
   return false;
}

/*
* Sets the position of the stage in steps
* I believe these are the microsteps of the device
*/
int SquidXYStage::SetPositionSteps(long xSteps, long ySteps)
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_MOVETO_X;
   long payLoad = 0;
   int numberOfBytes = 4;
   if (xSteps >= 0)
      payLoad = xSteps;
   else
      //  payLoad = 2**(8 * 4) + xSteps; //find two's completement
      payLoad = xSteps;
   cmd[2] = xSteps >> 24;
   cmd[3] = (xSteps >> 16) & 0xFF;
   cmd[4] = (xSteps >> 8) & 0xFF;
   cmd[5] = xSteps & 0xFF;

   int ret = hub_->SendCommand(cmd, cmdSize, &cmdNr_);
   if (ret != DEVICE_OK)
      return ret;
   changedTime_ = GetCurrentMMTime();

   return DEVICE_OK;
}


