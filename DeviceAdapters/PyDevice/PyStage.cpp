#include "pch.h"
#include "PyStage.h"

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
    pos = object_.Get("position").as<double>() - home_;
    return CheckError();
}

int CPyXYStage::GetPositionUm(double& x, double& y) {
    _check_(Wait());
    x = object_.Get("position_x").as<double>() - home_x_;
    y = object_.Get("position_y").as<double>() - home_y_;
    return CheckError();
}

int CPyStage::SetPositionUm(double pos) {
    object_.Set("position", pos + home_);
    _check_(Wait());
    OnStagePositionChanged(pos);
    return CheckError();
}

int CPyXYStage::SetPositionUm(double x, double y) {
    object_.Set("position_x", x + home_x_);
    object_.Set("position_y", y + home_y_);
    _check_(Wait());
    OnXYStagePositionChanged(x, y);
    return CheckError();
}

int CPyXYStage::Stop() {
    // similar to GetPositionUm, but don't wait for the stage to reach the end point
    double x = object_.Get("position_x").as<double>() - home_x_;
    double y = object_.Get("position_y").as<double>() - home_y_;
    return SetPositionUm(x, y);
}

double CPyStage::StepSizeUm() const {
    return object_.Get("step_size").as<double>();
}

double CPyXYStage::GetStepSizeXUm() {
    return object_.Get("step_size_x").as<double>();
}

double CPyXYStage::GetStepSizeYUm() {
    return object_.Get("step_size_y").as<double>();
}

// Sets current position as home
int CPyStage::SetOrigin() {
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
    _check_(GetPropertyLowerLimit("PositionX", x_lower));
    _check_(GetPropertyUpperLimit("PositionX", x_upper));
    _check_(GetPropertyLowerLimit("PositionY", y_lower));
    _check_(GetPropertyUpperLimit("PositionY", y_upper));
    x_lower -= home_x_;
    x_upper -= home_x_;
    y_lower -= home_y_;
    y_upper -= home_y_;
    return x_upper == x_lower ? DEVICE_UNSUPPORTED_COMMAND : DEVICE_OK;

}
