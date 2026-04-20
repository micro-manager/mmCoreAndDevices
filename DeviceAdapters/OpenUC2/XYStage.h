#ifndef _OPENUC2_XYSTAGE_H_
#define _OPENUC2_XYSTAGE_H_

#include "DeviceBase.h"
#include "openuc2.h"
#include <string>

class UC2Hub;

class XYStage : public CXYStageBase<XYStage>
{
public:
   XYStage();
   ~XYStage();

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   // Required XYStage API overrides
   double GetStepSizeXUm() override { return stepSizeUm_; }
   double GetStepSizeYUm() override { return stepSizeUm_; }
   int SetPositionSteps(long x, long y) override;
   int GetPositionSteps(long& x, long& y) override;
   int SetRelativePositionSteps(long x, long y) override;
   int Home() override;
   int Stop() override;
   int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) override;
   int IsXYStageSequenceable(bool& isSequenceable) const override;
   // Additional methods required by MM::XYStage:
   int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) override;
   int SetOrigin() override;

private:
   bool     initialized_;
   UC2Hub*  hub_;
   long     posXSteps_;
   long     posYSteps_;
   double   stepSizeUm_;
};

#endif
