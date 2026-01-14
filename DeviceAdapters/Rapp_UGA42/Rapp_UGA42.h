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
#include "DeviceThreads.h"
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>

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

// Forward declarations
class RappUGA42Scanner;
class DeviceWorkerThread;

// Command pattern for async device operations
class Command
{
public:
   enum Type {
      POINT_AND_FIRE,
      SET_POSITION,
      SET_ILLUMINATION,
      RUN_POLYGONS,
      LOAD_POLYGONS
   };

   virtual ~Command() {}
   virtual Type GetType() const = 0;
   virtual int Execute(RappUGA42Scanner& device) = 0;
};

class PointAndFireCommand : public Command
{
public:
   PointAndFireCommand(double x, double y, double pulseTime_us) :
      x_(x), y_(y), pulseTime_us_(pulseTime_us) {}

   Type GetType() const { return POINT_AND_FIRE; }
   int Execute(RappUGA42Scanner& device);

private:
   double x_;
   double y_;
   double pulseTime_us_;
};

class SetPositionCommand : public Command
{
public:
   SetPositionCommand(double x, double y) : x_(x), y_(y) {}

   Type GetType() const { return SET_POSITION; }
   int Execute(RappUGA42Scanner& device);

private:
   double x_;
   double y_;
};

class SetIlluminationCommand : public Command
{
public:
   SetIlluminationCommand(bool on) : on_(on) {}

   Type GetType() const { return SET_ILLUMINATION; }
   int Execute(RappUGA42Scanner& device);

private:
   bool on_;
};

class RunPolygonsCommand : public Command
{
public:
   RunPolygonsCommand(const std::vector<std::vector<RPOINTF>>& polygons, int repetitions) :
      polygons_(polygons), repetitions_(repetitions) {}

   Type GetType() const { return RUN_POLYGONS; }
   int Execute(RappUGA42Scanner& device);

private:
   std::vector<std::vector<RPOINTF>> polygons_;
   int repetitions_;
};

class LoadPolygonsCommand : public Command
{
public:
   LoadPolygonsCommand(const std::vector<std::vector<RPOINTF>>& polygons) :
      polygons_(polygons) {}

   Type GetType() const { return LOAD_POLYGONS; }
   int Execute(RappUGA42Scanner& device);

private:
   std::vector<std::vector<RPOINTF>> polygons_;
};

// Worker thread for async device operations
class DeviceWorkerThread : public MMDeviceThreadBase
{
public:
   DeviceWorkerThread(RappUGA42Scanner& device);
   ~DeviceWorkerThread();

   int svc();
   int open(void*) { return 0; }
   int close(unsigned long) { return 0; }

   void Start();
   void Stop();
   void EnqueueCommand(Command* cmd);
   bool IsBusy();
   void StopPolygonSequence();

private:
   RappUGA42Scanner& device_;
   std::queue<Command*> commandQueue_;
   std::mutex queueMutex_;
   std::condition_variable queueCV_;
   bool stop_;
   Command* activeCommand_;
   bool stopPolygonRequested_;
   std::chrono::steady_clock::time_point lastKeepaliveTime_;

   // Prevent copying - thread objects should not be copied
   DeviceWorkerThread(const DeviceWorkerThread&) = delete;
   DeviceWorkerThread& operator=(const DeviceWorkerThread&) = delete;
};

class RappUGA42Scanner : public CGalvoBase<RappUGA42Scanner>
{
public:
   RappUGA42Scanner();
   ~RappUGA42Scanner();

   // Friend classes for command execution
   friend class PointAndFireCommand;
   friend class SetPositionCommand;
   friend class SetIlluminationCommand;
   friend class RunPolygonsCommand;
   friend class LoadPolygonsCommand;
   friend class DeviceWorkerThread;

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
   int OnDebugMode(MM::PropertyBase* pProp, MM::ActionType eAct);
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
   UINT16 loadedPolygonSequenceID_;  // Cached polygon sequence from LoadPolygons
   bool polygonsLoaded_;              // Flag indicating polygons are pre-uploaded

   // Worker thread for async operations
   DeviceWorkerThread* workerThread_;

   // Helper functions
   // ----------------
   int MapRetCode(UINT32 retCode);
   int AddLaser();
   int UpdateLaser();
   State GetDeviceState();
   bool IsDeviceIdle();
   bool Reconnect();
};

#endif //_Rapp_UGA42_H_
