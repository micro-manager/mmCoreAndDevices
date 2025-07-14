///////////////////////////////////////////////////////////////////////////////
// FILE:          AndorLaserCombiner.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   AndorLaserCombiner controller adapter
// COPYRIGHT:     University of California, San Francisco, 2009
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Karl Hoover, UCSF
//
//

#ifndef _AndorLaserCombiner_H_
#define _AndorLaserCombiner_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceUtils.h"
#include <string>
//#include <iostream>
#include <vector>
//using namespace std;
const int MaxLasers = 10;

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
class ALCImpl;
class AndorLaserCombiner;

class PiezoStage : public CStageBase<PiezoStage>
{

   friend class AndorLaserCombiner;

public:

   PiezoStage( const char* name);
   ~PiezoStage();

   // Stage API
   // ---------
   int SetPositionUm(double pos);
   int GetPositionUm(double& pos);
   int SetPositionSteps(long steps);
   int GetPositionSteps(long& steps);
   int SetOrigin();
   int GetLimits(double& min, double& max);

   int SetRelativePositionUm(double);

   void GetName(char* Name) const;
   int Initialize();
   int Shutdown();
   bool Busy();

   int OnPiezoRange(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPiezoPosition(MM::PropertyBase* pProp, MM::ActionType eAct);

   bool IsContinuousFocusDrive() const {return false;}

   // TODO: Implement these for Andor Laser Z sequencing
   // Sequence functions
   int IsStageSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}
   int GetStageSequenceMaxLength(long& nrEvents) const  {nrEvents = 0; return DEVICE_OK;}
   int StartStageSequence()  {return DEVICE_OK;}
   int StopStageSequence()  {return DEVICE_OK;}
   int ClearStageSequence() {return DEVICE_OK;}
   int AddToStageSequence(double /* position */) {return DEVICE_OK;}
   int SendStageSequence()  {return DEVICE_OK;}

private:

   /** Implementation instance shared with AndorLaserCombiner. */
	ALCImpl* pImpl_;

   std::string name_;
	float PiezoRange(void);
	void PiezoRange(const float);
	float PiezoPosition(void);
	void PiezoPosition(const float);
};


class AndorLaserCombiner : public CShutterBase<AndorLaserCombiner>
{

   friend class PiezoStage;

private:

   double minlp_;
   double maxlp_;

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

   /** Implementation instance shared with PiezoStage. */
   ALCImpl* pImpl_;
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
