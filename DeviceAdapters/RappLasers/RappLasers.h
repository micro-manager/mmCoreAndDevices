///////////////////////////////////////////////////////////////////////////////
// FILE:          RappLasers.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Rapp Laser Controller adapter
// COPYRIGHT:     University of California, San Francisco, 2026
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
// AUTHOR:        Nico Stuurman, n.stuurman@ucsf.edu, 01/12/2026

#ifndef _RAPPLASERS_H_
#define _RAPPLASERS_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <string>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN    10001
#define ERR_NO_PORT_SET              10002
#define ERR_DEVICE_NOT_FOUND         10003
#define ERR_COMMUNICATION            10004
#define ERR_INVALID_RESPONSE         10005
#define ERR_COMMAND_FAILED           10006

// Forward declaration
class PollingThread;

/**
 * RappLaser - Device adapter for Rapp laser controllers
 *
 * Controls Rapp lasers via binary serial protocol
 * Implements continuous polling to keep laser alive
 */
class RappLaser : public CGenericBase<RappLaser>
{
public:
   RappLaser();
   ~RappLaser();

   friend class PollingThread;

   // MMDevice API
   int Initialize();
   int Shutdown();
   void GetName(char* name) const;
   bool Busy() { return false; }

   // Property action handlers
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLight(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserName(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Thread-safe access
   static MMThreadLock& GetLock() { return lock_; }

   // Called by polling thread
   int QueryStatus();

private:
   // Device state
   std::string port_;
   bool initialized_;
   double answerTimeoutMs_;

   // Cached device state (updated by polling thread)
   bool shutterOpen_;
   bool lightOn_;
   double intensityPercent_;  // 0-100%
   std::string serialNumber_;
   std::string laserName_;

   // Threading
   static MMThreadLock lock_;
   PollingThread* pollingThread_;

   // Protocol commands
   static const unsigned char CMD_SERIAL_NUMBER = 0x19;
   static const unsigned char CMD_LASER_NAME = 0x1C;
   static const unsigned char CMD_SHUTTER = 0x42;
   static const unsigned char CMD_LIGHT = 0x43;
   static const unsigned char CMD_INTENSITY = 0x44;
   static const unsigned char CMD_STATUS = 0x40;
   static const unsigned char RESPONSE_OK = 0xFF;

   // Protocol implementation
   int SendInitializationSequence();
   int SendCommand(unsigned char cmd);
   int SendCommand(unsigned char cmd, unsigned char param);
   int SendCommand(unsigned char cmd, const unsigned char* params, int paramLen);
   int ReadResponse(unsigned char* response, int maxLen, unsigned long& bytesRead);
   int ReadStringResponse(std::string& response);
   int ReadFixedResponse(unsigned char* response, int expectedLen, unsigned long& bytesRead);

   // High-level device commands
   int SetShutterState(bool open);
   int GetShutterState(bool& open);
   int SetLightState(bool on);
   int GetLightState(bool& on);
   int SetIntensity(double intensityPercent);
   int GetIntensity(double& intensityPercent);
   int QuerySerialNumber(std::string& serialNum);
   int QueryLaserName(std::string& laserName);

   // Utility methods
   long PercentageToRaw(double percentage);
   double RawToPercentage(long rawValue);
   void IntensityToBytes(long intensity, unsigned char* bytes);
   long BytesToIntensity(const unsigned char* bytes);
};

/**
 * PollingThread - Background thread for continuous status polling
 *
 * Sends 0x40 status queries every 100ms to keep laser alive
 * and update cached intensity value
 */
class PollingThread : public MMDeviceThreadBase
{
public:
   PollingThread(RappLaser& device);
   ~PollingThread();

   int svc();
   int open(void*) { return 0; }
   int close(unsigned long) { return 0; }

   void Start();
   void Stop() { stop_ = true; }

private:
   RappLaser& device_;
   bool stop_;

   PollingThread& operator=(const PollingThread&) {
      return *this;
   }
};

#endif // _RAPPLASERS_H_
