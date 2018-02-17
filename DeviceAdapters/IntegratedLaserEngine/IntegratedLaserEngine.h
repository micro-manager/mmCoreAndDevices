///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _INTEGRATEDLASERENGINE_H_
#define _INTEGRATEDLASERENGINE_H_

#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/DeviceUtils.h"
#include <string>
#include <vector>
const int MaxLasers = 10;

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
class CILEWrapper;

class CIntegratedLaserEngine : public CShutterBase<CIntegratedLaserEngine>
{
public:
   // Power setting limits:
   double minlp(){ return minlp_;};
   void minlp(double v_a) { minlp_= v_a;};
   double maxlp(){ return maxlp_;};
   void maxlp(double v_a) { maxlp_= v_a;};

   CIntegratedLaserEngine( const char* name);
   ~CIntegratedLaserEngine();

   // MMDevice API.
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Action interface.
   int OnPowerSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnSaveLifetime(MM::PropertyBase* pProp, MM::ActionType eAct, long index);

   // Read-only properties.
   int OnNLasers(MM::PropertyBase* , MM::ActionType );
   int OnHours(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnIsLinear(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnMaximumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnWaveLength(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnLaserState(MM::PropertyBase* , MM::ActionType , long );
   int OnEnable(MM::PropertyBase* pProp, MM::ActionType, long index);

   // Shutter API.
   int SetOpen(bool open = true);
   int GetOpen(bool& open);
   int Fire(double deltaT);

   int Wavelength(const int laserIndex_a);  // nano-meters.
   int PowerFullScale(const int laserIndex_a);  // Unitless.  TODO Should be percentage IFF isLinear_.
   bool Ready(const int laserIndex_a);
   float PowerReadback(const int laserIndex_a);  // milli-Watts.
   bool AllowsExternalTTL(const int laserIndex_a);
   float PowerSetpoint(const int laserIndex_a);  // milli-Watts.
   void PowerSetpoint( const int laserIndex_a, const float);  // milli-Watts.

private:   
  double minlp_;
  double maxlp_;


   /** Implementation instance shared with PiezoStage. */
   CILEWrapper* pImpl_;
      // todo -- can move these to the implementation
   int HandleErrors();
   CIntegratedLaserEngine& operator = (CIntegratedLaserEngine& /*rhs*/)
   {
      assert(false);
      return *this;
   }

   void GenerateALCProperties();
   void GenerateReadOnlyIDProperties();

   int error_;
   bool initialized_;
   std::string name_;
   long armState_;
   bool busy_;
   double answerTimeoutMs_;
   MM::MMTime changedTime_;
   int nLasers_;
   float powerSetPoint_[MaxLasers+1];  // 1-based arrays therefore +1
   bool isLinear_[MaxLasers+1];
   std::string enable_[MaxLasers+1];
   std::vector<std::string> enableStates_[MaxLasers+1];
   enum EXTERNALMODE
   {
      CW,
      TTL_PULSED
   };
   std::string savelifetime_[MaxLasers+1];
   std::vector<std::string> savelifetimeStates_[MaxLasers+1];
   bool openRequest_;
   unsigned char laserPort_;  // First two bits of DOUT (0 or 1 or 2) IFF multiPortUnitPresent_
};


#endif // _AndorLaserCombiner_H_
