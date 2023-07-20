#pragma once
#include "PyDevice.h"

using PyStageClass = CPyDeviceTemplate<CStageBase<std::monostate>>;
class CPyStage : public PyStageClass {
public:
    CPyStage(const string& id) : PyStageClass(id) {}
    int Home() override;
    int GetPositionUm(double& pos) override;
    int SetPositionUm(double pos) override;
    int SetPositionSteps(long steps) override {
        return SetPositionUm(steps * StepSizeUm());
    }
    int GetPositionSteps(long& steps) override {
        double pos;
        _check_(GetPositionUm(pos));
        steps = std::lround((pos + home_) / StepSizeUm());
        return CheckError();
    }
    int SetOrigin() override;
    int SetAdapterOriginUm(double value) {
        home_ = value;
        return DEVICE_OK;
    }
    int GetLimits(double& lower, double& upper) override;
    int IsStageSequenceable(bool& value) const override {
        value = false;
        return DEVICE_OK;
    }
    bool IsContinuousFocusDrive() const override {
        return true;
    }
protected:
    int Wait();
    double StepSizeUm() const;
    double home_ = 0.0;
};

// note: we don't derive from CXYStageBase, because we don't use that functionality and prefer to keep track of position in micrometers.
using PyXYStageClass = CPyDeviceTemplate<CDeviceBase<MM::XYStage, std::monostate>>;
class CPyXYStage : public PyXYStageClass {
public:
    CPyXYStage(const string& id) : PyXYStageClass(id) {}
    int Home() override;
    int GetPositionUm(double& x, double& y) override;
    int SetPositionUm(double x, double y) override;
    int SetPositionSteps(long x, long y) override {
        return SetPositionUm((x + home_x_) * GetStepSizeXUm(), (y + home_y_) * GetStepSizeYUm());
    }
    int SetRelativePositionUm(double x, double y) override {
        double cx, cy;
        _check_(GetPositionUm(cx, cy));
        cx += x;
        cy += y;
        return SetPositionUm(x, y);
    }
    int SetRelativePositionSteps(long x, long y) override {
        long cx, cy;
        _check_(GetPositionSteps(cx, cy));
        cx += x;
        cy += y;
        return SetPositionSteps(x, y);
    }
    int SetOrigin() override;
    int SetXOrigin() override;
    int SetYOrigin() override;
    int SetAdapterOriginUm(double x, double y) {
        home_x_ = x;
        home_y_ = y;
        return DEVICE_OK;
    }
    int GetLimitsUm(double& x_lower, double& x_upper, double& y_lower, double& y_upper) override;
    int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) override {
        double x_lower, x_upper, y_lower, y_upper;
        _check_(GetLimitsUm(x_lower, x_upper, y_lower, y_upper));
        double x_step = GetStepSizeXUm();
        double y_step = GetStepSizeXUm();
        xMin = std::lround((x_lower + home_x_) / x_step);
        xMax = std::lround((x_lower + home_x_) / x_step);
        yMin = std::lround((y_lower + home_y_) / y_step);
        yMax = std::lround((y_lower + home_y_) / y_step);
        return DEVICE_OK;
    };
    int GetPositionSteps(long& x_steps, long& y_steps) override {
        double x, y;
        _check_(GetPositionUm(x, y));
        x_steps = std::lround((x + home_x_) / GetStepSizeXUm());
        y_steps = std::lround((y + home_y_) / GetStepSizeYUm());
        return CheckError();
    }
    int Stop() override;
    int IsXYStageSequenceable(bool& value) const override {
        value = false;
        return DEVICE_OK;
    }
    double GetStepSizeXUm() override;
    double GetStepSizeYUm() override;
    virtual int Move(double /*x_velocity*/, double /*y_velocity*/) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int GetXYStageSequenceMaxLength(long& /*nrEvents*/) const {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int StartXYStageSequence() {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int StopXYStageSequence() {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int ClearXYStageSequence() {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int AddToXYStageSequence(double /*positionX*/, double /*positionY*/) {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    virtual int SendXYStageSequence() {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

protected:
    int Wait();
    double home_x_ = 0.0;
    double home_y_ = 0.0;
};