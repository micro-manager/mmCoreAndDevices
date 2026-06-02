///////////////////////////////////////////////////////////////////////////////
// FILE:          EnderscopeStage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//
// DESCRIPTION:   Enderscope Stage adapter (Marlin/Enderscope-compatible)
//                Inspired by adapter structure used in Marzhauser-LStep.
///////////////////////////////////////////////////////////////////////////////

#ifndef _ENDERSCOPE_STAGE_H_
#define _ENDERSCOPE_STAGE_H_

#include "DeviceBase.h"
#include "MMDeviceConstants.h"
#include <string>

extern const char* g_EnderscopeXYStageDeviceName;
extern const char* g_EnderscopeZStageDeviceName;

class EnderscopeBase
{
public:
   explicit EnderscopeBase(MM::Device* device);
   virtual ~EnderscopeBase();

protected:
   int CheckDeviceStatus();
   int ClearPort();
   int SendCommand(const std::string& command) const;
   int ReadLine(std::string& line) const;
   int CommandExpectOk(const std::string& command) const;
   int QueryPositionMm(double& x, double& y, double& z) const;

   static std::string Trim(const std::string& input);
   static bool ParseAxisValue(const std::string& line, char axis, double& value);

protected:
   bool initialized_;
   std::string port_;
   long baudRate_;
   long readTimeoutMs_;

   MM::Device* device_;
   MM::Core* core_;
};

class EnderscopeXYStage : public CXYStageBase<EnderscopeXYStage>, public EnderscopeBase
{
public:
   EnderscopeXYStage();
   ~EnderscopeXYStage() override;

   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   int SetPositionUm(double x, double y) override;
   int GetPositionUm(double& x, double& y) override;
   int SetRelativePositionUm(double dx, double dy) override;
   int SetPositionSteps(long x, long y) override;
   int GetPositionSteps(long& x, long& y) override;
   int SetRelativePositionSteps(long x, long y) override;
   int Home() override;
   int Stop() override;
   int SetOrigin() override;
   int SetAdapterOriginUm(double x, double y) override;
   int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) override;
   int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) override;

   int IsXYStageSequenceable(bool& isSequenceable) const override
   {
      isSequenceable = false;
      return DEVICE_OK;
   }

   double GetStepSizeXUm() override { return stepSizeXUm_; }
   double GetStepSizeYUm() override { return stepSizeYUm_; }

   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBaudRate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReadTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int SetAbsoluteMm(double xMm, double yMm);
   int SetRelativeMm(double dxMm, double dyMm);

   double stepSizeXUm_;
   double stepSizeYUm_;
   double originXUm_;
   double originYUm_;

   mutable double lastXUm_;
   mutable double lastYUm_;
};

class EnderscopeZStage : public CStageBase<EnderscopeZStage>, public EnderscopeBase
{
public:
   EnderscopeZStage();
   ~EnderscopeZStage() override;

   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   int SetPositionUm(double pos) override;
   int SetRelativePositionUm(double d) override;
   int GetPositionUm(double& pos) override;
   int SetPositionSteps(long steps) override;
   int GetPositionSteps(long& steps) override;
   int Stop() override;
   int Home() override;
   int SetOrigin() override;
   int SetAdapterOriginUm(double d) override;
   int GetLimits(double& min, double& max) override;

   int IsStageSequenceable(bool& isSequenceable) const override
   {
      isSequenceable = false;
      return DEVICE_OK;
   }

   bool IsContinuousFocusDrive() const override { return false; }

   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBaudRate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReadTimeout(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStepSize(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int SetAbsoluteMm(double zMm);
   int SetRelativeMm(double dzMm);

   double stepSizeUm_;
   double originZUm_;

   mutable double lastZUm_;
};

#endif // _ENDERSCOPE_STAGE_H_
