///////////////////////////////////////////////////////////////////////////////
// FILE:          Rapp_UGA42.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Rapp UGA-42 Scanner adapter
//
// COPYRIGHT:     University of California, San Francisco
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER(S) OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nico Stuurman, 2025
//
// Based on the Rapp UGA-40 adapter by Arthur Edelstein

#ifndef _Rapp_UGA42_H_
#define _Rapp_UGA42_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <vector>
#include <map>

#pragma warning( push )
#pragma warning( disable : 4251 )
#include "ROE UGA-42.h"
#pragma warning( pop )

// Error codes
#define ERR_PORT_CHANGE_FORBIDDEN    10001
#define ERR_DEVICE_NOT_FOUND         10002
#define ERR_CONNECTION_FAILED        10003
#define ERR_LASER_SETUP_FAILED       10004
#define ERR_SEQUENCE_UPLOAD_FAILED   10005
#define ERR_SEQUENCE_START_FAILED    10006
#define ERR_INVALID_DEVICE_STATE     10007
#define ERR_MEMORY_OVERLOAD          10008
#define ERR_SEQUENCE_INVALID         10009

class RappUGA42Scanner : public CGalvoBase<RappUGA42Scanner>
{
public:
   RappUGA42Scanner();
   ~RappUGA42Scanner();

   // Device API
   // ----------
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Galvo API
   // ---------
   int PointAndFire(double x, double y, double pulseTime_us);
   int SetSpotInterval(double pulseTime_us);
   int SetPosition(double x, double y);
   int GetPosition(double& x, double& y);
   int SetIlluminationState(bool on);
   int AddPolygonVertex(int polygonIndex, double x, double y);
   int DeletePolygons();
   int LoadPolygons();
   int SetPolygonRepetitions(int repetitions);
   int RunPolygons();
   int RunSequence();
   int StopSequence();
   int GetChannel(char* channelName);

   double GetXRange();
   double GetYRange();

   // Property action handlers
   // ------------------------
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpotSize(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTTLTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTTLTriggerBehavior(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserType(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserFrequency(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTickTime(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnDigitalRiseTime(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnDigitalFallTime(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnAnalogChangeTime(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMinIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   // Device connection
   bool initialized_;
   std::string port_;
   UGA42* device_;
   bool debugMode_;

   // Laser configuration
   UINT32 laserID_;
   bool laserAdded_;
   OutputPorts laserPort_;
   UINT32 digitalRiseTime_;   // microseconds
   UINT32 digitalFallTime_;   // microseconds
   UINT32 analogChangeTime_;  // microseconds
   UINT32 minIntensity_;      // 0-10000
   UINT32 maxIntensity_;      // 0-10000
   UINT32 laserFrequency_;    // Hz, 0 for continuous
   UINT32 currentIntensity_;  // 0-10000
   int spotSize_;             // device coordinates

   // Timing
   UINT32 tickTime_;          // microseconds
   double pulseTime_us_;

   // Scanning mode
   ScanMode scanMode_;

   // TTL triggering
   InputPorts ttlTriggerPort_;
   TriggerBehaviour ttlTriggerBehavior_;

   // Position tracking
   double currentX_;
   double currentY_;

   // Polygon support
   std::vector<std::vector<RPOINTF>> polygons_;
   int polygonRepetitions_;

   // Sequence management
   UINT16 lastSequenceID_;
   bool sequenceRunning_;

   // Helper functions
   // ----------------
   int MapRetCode(UINT32 retCode);
   int AddLaser();
   int UpdateLaser();
   State GetDeviceState();
   bool IsDeviceIdle();
};

#endif //_Rapp_UGA42_H_
