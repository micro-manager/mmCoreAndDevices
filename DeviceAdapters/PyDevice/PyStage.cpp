#include "pch.h"
#include "PyStage.h"

inline PyObj to_um(double value) {
    return PyObj(value) * CPyHub::g_unit_um;
}
inline double from_um(const PyObj& value) {
    return (value / CPyHub::g_unit_um).as<double>();
}

int CPyStage::Home()
{
    object_.CallMember("home");
    home_ = 0.0;
    OnStagePositionChanged(0.0);
    return CheckError();
}

int CPyXYStage::Home()
{
    object_.CallMember("home");
    home_x_ = 0.0;
    home_y_ = 0.0;
    OnXYStagePositionChanged(0.0, 0.0);
    return CheckError();
}

int CPyStage::Wait() {
    object_.CallMember("wait");
    return CheckError();
}

int CPyXYStage::Wait() {
    object_.CallMember("wait");
    return CheckError();
}

int CPyStage::GetPositionUm(double& pos) {
    _check_(Wait());
    pos = from_um(object_.Get("position")) - home_;
    return CheckError();
}

int CPyXYStage::GetPositionUm(double& x, double& y) {
    PyLock lock;
    _check_(Wait());
    x = from_um(object_.Get("x")) - home_x_;
    y = from_um(object_.Get("y")) - home_y_;
    return CheckError();
}

int CPyStage::SetPositionUm(double pos) {
    PyLock lock;
    object_.Set("position", to_um(pos + home_));
    _check_(Wait());
    OnStagePositionChanged(pos);
    return CheckError();
}

int CPyXYStage::SetPositionUm(double x, double y) {
    PyLock lock;
    object_.Set("x", to_um(x + home_x_));
    object_.Set("y", to_um(y + home_y_));
    _check_(Wait());
    OnXYStagePositionChanged(x, y);
    return CheckError();
}

int CPyXYStage::Stop() {
    // similar to GetPositionUm, but don't wait for the stage to reach the end point
    PyLock lock;
    double x = from_um(object_.Get("x")) - home_x_;
    double y = from_um(object_.Get("y")) - home_y_;
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

// Sets current position as home
int CPyStage::SetOrigin() {
    PyLock lock;
    _check_(Wait());
    return GetPositionUm(home_);
}

// Sets current position as home
int CPyXYStage::SetXOrigin() {
    _check_(Wait());
    double dummy;
    return GetPositionUm(home_x_, dummy);
}

int CPyXYStage::SetYOrigin() {
    _check_(Wait());
    double dummy;
    return GetPositionUm(dummy, home_y_);
}

int CPyXYStage::SetOrigin() {
    _check_(Wait());
    return GetPositionUm(home_x_, home_y_);
}

int CPyStage::GetLimits(double& lower, double& upper) {
    _check_(Wait());
    _check_(GetPropertyLowerLimit("Position", lower));
    _check_(GetPropertyUpperLimit("Position", upper));
    lower -= home_;
    upper -= home_;
    return upper == lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;
}

int CPyXYStage::GetLimitsUm(double& x_lower, double& x_upper, double& y_lower, double& y_upper) {
    _check_(Wait());
    _check_(GetPropertyLowerLimit("X", x_lower));
    _check_(GetPropertyUpperLimit("X", x_upper));
    _check_(GetPropertyLowerLimit("Y", y_lower));
    _check_(GetPropertyUpperLimit("Y", y_upper));
    x_lower -= home_x_;
    x_upper -= home_x_;
    y_lower -= home_y_;
    y_upper -= home_y_;
    return x_upper == x_lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;

}
