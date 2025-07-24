#ifndef _OPENUC2_ZSTAGE_H_
#define _OPENUC2_ZSTAGE_H_

#include "DeviceBase.h"
#include "openuc2.h"
#include <string>

class UC2Hub;

class ZStage : public CStageBase<ZStage>
{
public:
   ZStage();
   ~ZStage();

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   // Required pure virtual methods from CStageBase
   int SetPositionSteps(long steps) override;
   int GetPositionSteps(long& steps) override;
   // Note: SetRelativePositionSteps is not declared in the base class;
   // we implement it as an additional helper (without override).
   virtual int SetRelativePositionSteps(long steps);
   int Home() override;
   int Stop() override;
   int GetLimits(double& min, double& max) override;

   // Methods required by MM::Stage interface:
   int SetPositionUm(double z) override;
   int GetPositionUm(double& z) override;
   int SetOrigin() override;
   int IsStageSequenceable(bool& isSequenceable) const override;
   bool IsContinuousFocusDrive() const override;

private:
   bool    initialized_;
   UC2Hub* hub_;

   long    posZSteps_;
   double  stepSizeUm_;
};

#endif
