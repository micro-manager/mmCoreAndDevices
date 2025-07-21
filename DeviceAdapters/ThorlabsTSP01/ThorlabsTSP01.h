///////////////////////////////////////////////////////////////////////////////
// FILE:          ThorlabsTSP01.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters - Thorlabs TSP01revB adapter
//-----------------------------------------------------------------------------
// DESCRIPTION:   This device adapter interfaces with Thorlabs thermometer TSP01-revB
//
//                
// AUTHOR:        Andrey Andreev, Ellison Medical Institute, 2025
//                aandreev@emila.org

#pragma once

#include <stdio.h>
#include "MMDevice.h"
#include "DeviceBase.h"
#include "visa.h"
#include "TLTSPB.h"
#include <thread>
#include <atomic>
#include <mutex>


#define ERR_NO_TSP01_CONNECTED      104


/*===========================================================================
 Prototypes
===========================================================================*/
void error_exit(ViSession instrHdl, ViStatus err);
void waitKeypress(void);

ViStatus get_device_id(ViSession ihdl);

ViStatus get_offset_values(ViSession ihdl);

ViStatus get_temperature_value(ViSession ihdl);
ViStatus get_humidity_value(ViSession ihdl);
ViStatus get_temperature_and_humidity_multi(ViSession ihdl);


class ThorlabsTSP01 : public CGenericBase<ThorlabsTSP01>
{
public:
	ThorlabsTSP01();
	~ThorlabsTSP01();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// action interface
	// ----------------
	int OnTSPName(MM::PropertyBase* pProp, MM::ActionType eAct);

	std::thread tempThread_;
	std::atomic<bool> keepRunning_;
	std::mutex tempMutex_;
	double latestTemp_;
	double latestHumidity_;
	double latestTemp_probe1_;
	double latestTemp_probe2_;

	void BackgroundTemperatureUpdate();  // thread function

private:
	bool initialized_;
	int FindResource(ViChar*& buffer);
	ViSession instrHdl_;
	MM::MMTime changedTime_;
	std::string deviceName_;
};
