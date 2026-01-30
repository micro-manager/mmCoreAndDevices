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
    int SetRelativePositionUm(double dpos) override;
    int GetPositionSteps(long& steps) override {
        double pos;
        _check_(GetPositionUm(pos));
        steps = std::lround((pos + origin_) / StepSizeUm());
        return CheckError();
    }
    int Stop() override;
    int SetOrigin() override;
    int SetAdapterOriginUm(double value) override {
        origin_ = value;
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
    double StepSizeUm() const;
    double origin_ = 0.0;
    double set_pos_ = NAN;
    PyObj home_;
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
        return SetPositionUm((x + origin_x_) * GetStepSizeXUm(), (y + origin_y_) * GetStepSizeYUm());
    }
    int SetRelativePositionUm(double dx, double dy) override;
    int SetRelativePositionSteps(long dx, long dy) override {
        return SetRelativePositionUm(dx * GetStepSizeXUm(), dy * GetStepSizeYUm());
    }
    int SetOrigin() override;
    int SetXOrigin() override;
    int SetYOrigin() override;
    int SetAdapterOriginUm(double x, double y) override {
        origin_x_ = x;
        origin_y_ = y;
        return DEVICE_OK;
    }
    int GetLimitsUm(double& x_lower, double& x_upper, double& y_lower, double& y_upper) override;
    int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) override {
        double x_lower, x_upper, y_lower, y_upper;
        _check_(GetLimitsUm(x_lower, x_upper, y_lower, y_upper));
        double x_step = GetStepSizeXUm();
        double y_step = GetStepSizeXUm();
        xMin = std::lround((x_lower + origin_x_) / x_step);
        xMax = std::lround((x_lower + origin_x_) / x_step);
        yMin = std::lround((y_lower + origin_y_) / y_step);
        yMax = std::lround((y_lower + origin_y_) / y_step);
        return DEVICE_OK;
    }
    int GetPositionSteps(long& x_steps, long& y_steps) override {
        double x, y;
        _check_(GetPositionUm(x, y));
        x_steps = std::lround((x + origin_x_) / GetStepSizeXUm());
        y_steps = std::lround((y + origin_y_) / GetStepSizeYUm());
        return CheckError();
    }
    int Stop() override;
    int IsXYStageSequenceable(bool& value) const override {
        value = false;
        return DEVICE_OK;
    }
    double GetStepSizeXUm() override;
    double GetStepSizeYUm() override;
    int Move(double /*x_velocity*/, double /*y_velocity*/) override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int GetXYStageSequenceMaxLength(long& /*nrEvents*/) const override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int StartXYStageSequence() override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int StopXYStageSequence() override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int ClearXYStageSequence() override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int AddToXYStageSequence(double /*positionX*/, double /*positionY*/) override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int SendXYStageSequence() override {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int UsesOnXYStagePositionChanged(bool& result) const override {
        result = false;
        return DEVICE_OK;
    }

protected:
    double origin_x_ = 0.0;
    double origin_y_ = 0.0;
    double set_pos_x_ = NAN;
    double set_pos_y_ = NAN;
    PyObj home_;
};