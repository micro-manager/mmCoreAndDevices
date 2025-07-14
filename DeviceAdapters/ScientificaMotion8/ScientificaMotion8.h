///////////////////////////////////////////////////////////////////////////////
// FILE:          ScientificaMotion8.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Scientifica Motion 8 rack adapter
// COPYRIGHT:     University of California, San Francisco, 2006
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
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/01/2006
//
//				  Scientifica Specific Parts
// AUTHOR:		  Matthew Player (ElecSoft Solutions)

#ifndef _SCIENTIFICA_MOTION_8_H_
#define _SCIENTIFICA_MOTION_8_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>
#include "ScientificaTxPacket.h"
#include "ScientificaRxPacket.h"

class ScientificaMotion8Hub : public HubBase<ScientificaMotion8Hub>
{
public:
	ScientificaMotion8Hub();
	~ScientificaMotion8Hub();

	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	bool Busy();

	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);
	int DetectInstalledDevices();

	int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);

	ScientificaRxPacket* WriteRead(ScientificaTxPacket* tx, int expected_length);
	bool CheckControllerVersion();
	int Stop(uint8_t device);
	int SetPosition(uint8_t device, uint8_t axis, long steps);
	int IsMoving(uint8_t device, bool* moving);
	bool initialized_;
private:
	std::string port_;
	static MMThreadLock lock_;
	int ReadControllerMap(void);
	

	uint8_t device_1_x_channel_;
	uint8_t device_1_y_channel_;
	uint8_t device_1_z_channel_;
	uint8_t device_1_f_channel_;

	uint8_t device_2_x_channel_;
	uint8_t device_2_y_channel_;
	uint8_t device_2_z_channel_;
	uint8_t device_2_f_channel_;
};


class M8XYStage : public CXYStageBase<M8XYStage>
{
public:
	M8XYStage(uint8_t device);
	~M8XYStage();

	bool Busy();
	void GetName(char* pName) const;

	int Initialize();
	int Shutdown();

	// XYStage API
	int SetPositionSteps(long x, long y);
	int GetPositionSteps(long& x, long& y);
	int Home();
	int Stop();
	int SetOrigin();
	int SetXOrigin();
	int SetYOrigin();
	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
	double GetStepSizeXUm() { return 0.01; }
	double GetStepSizeYUm() { return 0.01; }
	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }
private:
	uint8_t device_;
	std::string name_;
};

class M8ZStage : public CStageBase<M8ZStage>
{
public:
	M8ZStage(uint8_t device);
	~M8ZStage();

	//Device API
	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	bool Busy();

	// Stage API
	int GetPositionUm(double& pos);
	int SetPositionUm(double pos);
	int SetPositionSteps(long steps);
	int GetPositionSteps(long& steps);
	int Home();
	int Stop();
	int SetOrigin();
	int GetLimits(double& min, double& max);
	int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }
	bool IsContinuousFocusDrive() const { return false; }
private:
	uint8_t device_;
	std::string name_;
};

class M8FilterCubeTurret : public CStateDeviceBase<M8FilterCubeTurret>
{
public:
	M8FilterCubeTurret(uint8_t device);
	~M8FilterCubeTurret();


	//Device API
	int Initialize();
	int Shutdown();
	void GetName(char* pName) const;
	bool Busy();

	unsigned long GetNumberOfPositions()const { return numPositions_; }

	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
private:
	uint8_t device_;
	std::string name_;

	int SetFilter(int filterIndex);
	int GetFilter(int& filterIndex);

	int numPositions_;
};

#endif
