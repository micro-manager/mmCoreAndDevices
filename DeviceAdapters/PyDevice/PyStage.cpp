#include "pch.h"
#include "PyStage.h"

inline PyObj to_um(double value) {
    return PyObj(value) * PyObj::g_unit_um;
}
inline double from_um(const PyObj& value) {
    return (value / PyObj::g_unit_um).as<double>();
}

/**
 * Home the stage
 *
 * For stages that support homing, the stage is moved to some fixed position (e.g. using a homing switch). This position is defined as position 0.0.
 * This function does _not_ wait for homing to complete, but does update the position known to MM ('OnStagePositionChanged'). Also see `Busy()`.
 * 
*/
int CPyStage::Home()
{
    object_.CallMember("home");
    origin_ = 0.0;
    set_pos_ = 0.0;
    OnStagePositionChanged(0.0);
    return CheckError();
}

int CPyXYStage::Home()
{
    object_.CallMember("home");
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
    auto retval = object_.CallMember("busy").as<bool>();
    CheckError();
    return retval;
}

bool CPyXYStage::Busy() {
    auto retval = object_.CallMember("busy").as<bool>();
    CheckError();
    return retval;
}

/**
 * Returns the current position of the stage in micrometers.
 * This function does _not_ wait for the stage to reach a stable position, so it may still be moving at this point.
*/
int CPyStage::GetPositionUm(double& pos) {
    pos = from_um(object_.Get("position")) - origin_;
    return CheckError();
}

int CPyXYStage::GetPositionUm(double& x, double& y) {
    PyLock lock;
    x = from_um(object_.Get("x")) - origin_x_;
    y = from_um(object_.Get("y")) - origin_y_;
    return CheckError();
}

/**
 * Sets the desired absolute position of the stage in micrometers.
 * This function does _not_ wait for the stage to reach this position.
*/
int CPyStage::SetPositionUm(double pos) {
    PyLock lock;
    object_.Set("position", to_um(pos + origin_));
    set_pos_ = pos;
    OnStagePositionChanged(pos);
    return CheckError();
}

int CPyXYStage::SetPositionUm(double x, double y) {
    PyLock lock;
    object_.Set("x", to_um(x + origin_x_));
    object_.Set("y", to_um(y + origin_y_));
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
    return from_um(object_.Get("step_size"));
}

double CPyXYStage::GetStepSizeXUm() {
    PyLock lock;
    return from_um(object_.Get("step_size_x"));
}

double CPyXYStage::GetStepSizeYUm() {
    PyLock lock;
    return from_um(object_.Get("step_size_y"));
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
    _check_(GetPropertyLowerLimit("Position", lower));
    _check_(GetPropertyUpperLimit("Position", upper));
    lower -= origin_;
    upper -= origin_;
    return upper == lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;
}

int CPyXYStage::GetLimitsUm(double& x_lower, double& x_upper, double& y_lower, double& y_upper) {
    _check_(GetPropertyLowerLimit("X", x_lower));
    _check_(GetPropertyUpperLimit("X", x_upper));
    _check_(GetPropertyLowerLimit("Y", y_lower));
    _check_(GetPropertyUpperLimit("Y", y_upper));
    x_lower -= origin_x_;
    x_upper -= origin_x_;
    y_lower -= origin_y_;
    y_upper -= origin_y_;
    return x_upper == x_lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;

}
