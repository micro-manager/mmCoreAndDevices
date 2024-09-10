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

SquidXYStage::SquidXYStage() :
   stepSize_um_(0.05),
   posX_um_(0.0),
   posY_um_(0.0),
   busy_(false)
{

}