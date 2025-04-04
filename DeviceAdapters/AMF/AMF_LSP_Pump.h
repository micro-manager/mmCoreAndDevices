///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_LSP_Pump.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for AMF LSP pumps.
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2024
//                
// VERSION:       0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    26.02.2024 LK

#ifndef _AMF_LSP_PUMP_H_
#define _AMF_LSP_PUMP_H_

#include "AMF_Commands.h"

#include "DeviceBase.h"
#include "DeviceThreads.h"


//////////////////////////////////////////////////////////////////////////////
// AMF_LSP_Pump class
// Device adapter for AMF LSP Pumps
//////////////////////////////////////////////////////////////////////////////

class AMF_LSP_Pump : public CVolumetricPumpBase<AMF_LSP_Pump>
{
public:
	AMF_LSP_Pump();
	~AMF_LSP_Pump();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	bool Busy();

    // MMPump API
    int Home();
    bool RequiresHoming() { return true; }
    int Stop();
    int GetMaxVolumeUl(double& volUl);
    int SetMaxVolumeUl(double volUl);
    int GetVolumeUl(double& volUl);
    int SetVolumeUl(double volUl);
    int IsDirectionInverted(bool& invert);
    int InvertDirection(bool invert);
    int GetFlowrateUlPerSecond(double& flowrate);
    int SetFlowrateUlPerSecond(double flowrate);
    int Start();
    int DispenseVolumeUl(double volUl);
    int DispenseDurationSeconds(double seconds);

    // Action Handlers
    int OnHome(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMaxVolume(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCurrentVolume(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFlowrate(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnRun(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    // Utility methods (internal use only)
    int SendRecv(AMF_Command cmd, long& value);
    double StepsToVolume(int steps);
    int VolumeToSteps(double volume);

    // AMF commands
    bool IsPumping();
    int MovePlunger(long position);
    int GetAddress(long& address);
    int GetNSteps(long& nSteps);
    int SetNSteps(long nSteps);

    // Communication class variables
    bool initialized_;
    bool busy_;
    std::string port_;
    long address_ = -1;
    std::string name_;

    // Pump state class variables
    bool isHomed_ = false;
    double minVolumeUl_ = 0;
    double maxVolumeUl_ = 1000;
    double volumeUl_ = 0;
    double stepVolumeUl_ = 0;
    double flowrateUlperSecond_ = 0;
    int currStep_ = 0;
    long nSteps_ = 3000;
    long run_ = 0;
};
#endif // _AMF_LSP_PUMP_H_
