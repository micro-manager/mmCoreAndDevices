#include "pch.h"
#include "PyStage.h"

const char* g_Keyword_Position = "Position-um";
const char* g_Keyword_X = "X-um";
const char* g_Keyword_Y = "Y-um";
const char* g_Keyword_StepSize = "StepSize-um";
const char* g_Keyword_StepSizeX = "StepSizeX-um";
const char* g_Keyword_StepSizeY = "StepSizeY-um";

/**
 * Home the stage
 *
 * For stages that support homing, the stage is moved to some fixed position (e.g. using a homing switch). This position is defined as position 0.0.
 * This function does _not_ wait for homing to complete, but does update the position known to MM ('OnStagePositionChanged'). Also see `Busy()`.
 * 
*/
int CPyStage::Home()
{
    home_.Call();
    origin_ = 0.0;
    set_pos_ = 0.0;
    OnStagePositionChanged(0.0);
    return CheckError();
}

int CPyXYStage::Home()
{
    home_.Call();
    origin_x_ = 0.0;
    origin_y_ = 0.0;
    set_pos_x_ = 0.0;
    set_pos_y_ = 0.0;
    OnXYStagePositionChanged(0.0, 0.0);
    return CheckError();
}

/**
 * Returns 'true' if the stage is still moving or settling
*/
bool CPyStage::Busy() {
    auto retval = busy_.Call().as<bool>();
    CheckError();
    return retval;
}

bool CPyXYStage::Busy() {
    auto retval = busy_.Call().as<bool>();
    CheckError();
    return retval;
}

/**
 * Returns the current position of the stage in micrometers.
 * This function does _not_ wait for the stage to reach a stable position, so it may still be moving at this point.
*/
int CPyStage::GetPositionUm(double& pos) {
    pos = GetFloatProperty(g_Keyword_Position) - origin_;
    return CheckError();
}

int CPyXYStage::GetPositionUm(double& x, double& y) {
    PyLock lock;
    x = GetFloatProperty(g_Keyword_X) - origin_x_;
    y = GetFloatProperty(g_Keyword_Y) - origin_y_;
    return CheckError();
}

/**
 * Sets the desired absolute position of the stage in micrometers.
 * This function does _not_ wait for the stage to reach this position.
*/
int CPyStage::SetPositionUm(double pos) {
    PyLock lock;
    SetFloatProperty(g_Keyword_Position, pos + origin_);
    set_pos_ = pos;
    OnStagePositionChanged(pos);
    return CheckError();
}

int CPyXYStage::SetPositionUm(double x, double y) {
    PyLock lock;
    SetFloatProperty(g_Keyword_X, x + origin_x_);
    SetFloatProperty(g_Keyword_Y, y + origin_y_);
    set_pos_x_ = x;
    set_pos_y_ = y;
    OnXYStagePositionChanged(x, y);
    return CheckError();
}

/**
 * Sets the position relative to the currently _set_ position (note, this may not be where the stage is at the moment if the stage is still moving)
*/
int CPyStage::SetRelativePositionUm(double dpos) {
    if (isnan(set_pos_))
        _check_(GetPositionUm(set_pos_));

    return SetPositionUm(set_pos_ + dpos);
}

int CPyXYStage::SetRelativePositionUm(double dx, double dy) {
    double dummy;
    if (isnan(set_pos_x_))
        _check_(GetPositionUm(set_pos_x_, dummy));
    if (isnan(set_pos_y_))
        _check_(GetPositionUm(dummy, set_pos_y_));

    return SetPositionUm(set_pos_x_ + dx, set_pos_y_ + dy);
}


/**
 * Stops the stage by ordering it to move to the current position
 * Does _not_ wait for the stage to stop. (see Busy())
*/
int CPyStage::Stop() {
    PyLock lock;
    double pos;
    _check_(GetPositionUm(pos));
    return SetPositionUm(pos);
}

int CPyXYStage::Stop() {
    PyLock lock;
    double x, y;
    _check_(GetPositionUm(x, y));
    return SetPositionUm(x, y);
}

double CPyStage::StepSizeUm() const {
    PyLock lock;
    return GetFloatProperty(g_Keyword_StepSize);
}

double CPyXYStage::GetStepSizeXUm() {
    PyLock lock;
    return GetFloatProperty(g_Keyword_StepSizeX);
}

double CPyXYStage::GetStepSizeYUm() {
    PyLock lock;
    return GetFloatProperty(g_Keyword_StepSizeY);
}

// Sets current position as home. Returns DEVICE_UNKNOWN_POSITION if the device is still moving
int CPyStage::SetOrigin() {
    if (Busy())
        return DEVICE_UNKNOWN_POSITION;
    return GetPositionUm(origin_);
}

// Sets current position as home
int CPyXYStage::SetXOrigin() {
    if (Busy())
        return DEVICE_UNKNOWN_POSITION;
    double dummy;
    return GetPositionUm(origin_x_, dummy);
}

int CPyXYStage::SetYOrigin() {
    if (Busy())
        return DEVICE_UNKNOWN_POSITION;
    double dummy;
    return GetPositionUm(dummy, origin_y_);
}

int CPyXYStage::SetOrigin() {
    if (Busy())
        return DEVICE_UNKNOWN_POSITION;
    return GetPositionUm(origin_x_, origin_y_);
}

int CPyStage::GetLimits(double& lower, double& upper) {
    _check_(GetPropertyLowerLimit(g_Keyword_Position, lower));
    _check_(GetPropertyUpperLimit(g_Keyword_Position, upper));
    lower -= origin_;
    upper -= origin_;
    return upper == lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;
}

int CPyXYStage::GetLimitsUm(double& x_lower, double& x_upper, double& y_lower, double& y_upper) {
    _check_(GetPropertyLowerLimit(g_Keyword_X, x_lower));
    _check_(GetPropertyUpperLimit(g_Keyword_X, x_upper));
    _check_(GetPropertyLowerLimit(g_Keyword_Y, y_lower));
    _check_(GetPropertyUpperLimit(g_Keyword_Y, y_upper));
    x_lower -= origin_x_;
    x_upper -= origin_x_;
    y_lower -= origin_y_;
    y_upper -= origin_y_;
    return x_upper == x_lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;
}
