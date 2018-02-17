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

class AndorLaserCombiner : public CShutterBase<AndorLaserCombiner>
{
public:
   // Power setting limits:
   double minlp(){ return minlp_;};
   void minlp(double v_a) { minlp_= v_a;};
   double maxlp(){ return maxlp_;};
   void maxlp(double v_a) { maxlp_= v_a;};

   AndorLaserCombiner( const char* name);
   ~AndorLaserCombiner();

   // MMDevice API.
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Action interface.
   int OnAddress(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPowerSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnSaveLifetime(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnConnectionType(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReceivedData(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Read-only properties.
   int OnNLasers(MM::PropertyBase* , MM::ActionType );
   int OnHours(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnIsLinear(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnMaximumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnWaveLength(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
   int OnLaserState(MM::PropertyBase* , MM::ActionType , long );
   int AndorLaserCombiner::OnEnable(MM::PropertyBase* pProp, MM::ActionType, long index);

   // Mechanical properties.
   int OnDIN(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnDOUT(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT1(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT2(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT3(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT4(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT5(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT6(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT7(MM::PropertyBase* pProp, MM::ActionType eAct);
   int AndorLaserCombiner::OnDOUT8(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMultiPortUnitPresent(MM::PropertyBase* , MM::ActionType );
   int OnLaserPort(MM::PropertyBase* , MM::ActionType );

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

   unsigned char DIN(void);
   void DOUT(const unsigned char);

private:   
  double minlp_;
  double maxlp_;


   /** Implementation instance shared with PiezoStage. */
   CILEWrapper* pImpl_;
      // todo -- can move these to the implementation
   int HandleErrors();
   AndorLaserCombiner& operator = (AndorLaserCombiner& /*rhs*/)
   {
      assert(false);
      return *this;
   }

   void GenerateALCProperties();
   void GenerateReadOnlyIDProperties();

   int error_;
   bool initialized_;
   std::string name_;
   unsigned char buf_[1000];
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
   unsigned char DOUT_;
   bool multiPortUnitPresent_;
   unsigned char laserPort_;  // First two bits of DOUT (0 or 1 or 2) IFF multiPortUnitPresent_
};


#endif // _AndorLaserCombiner_H_