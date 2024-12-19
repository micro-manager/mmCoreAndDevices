#pragma once

#include "DeviceBase.h"
#include "MMDevice.h"
#include <mutex>

#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109

const unsigned char cmd_version = 0;
const unsigned char cmd_start = 1;
const unsigned char cmd_stop = 2;
const unsigned char cmd_interval = 3; // interval in microseconds
const unsigned char cmd_pulse_duration = 4; // in microsconds
const unsigned char cmd_wait_for_input = 5;
const unsigned char cmd_number_of_pulses = 6;


/**
 * CameraSnapThread: helper thread
 */
class CameraSnapThread : public MMDeviceThreadBase
{
public:
   CameraSnapThread() :
      camera_(0),
      started_(false)
   {}

   ~CameraSnapThread() { if (started_) wait(); }

   void SetCamera(MM::Camera* camera) { camera_ = camera; }

   int svc() { camera_->SnapImage(); return 0; }

   void Start() { activate(); started_ = true; }

private:
   MM::Camera* camera_;
   bool started_;
};


class TeensyCom
{
public:

   TeensyCom(MM::Core* callback, MM::Device* device, const char* portLabel);

   ~TeensyCom() {};

   int SendCommand(uint8_t cmd, uint32_t param);
   int Enquire(uint8_t cmd);
   int GetRunningStatus(uint32_t& status);
   int GetResponse(uint8_t cmd, uint32_t& param);
   int GetVersion(uint32_t& version);
   int GetInterval(uint32_t& interval);
   int GetPulseDuration(uint32_t& pulseDuration);
   int GetWaitForInput(uint32_t& waitForInput);
   int GetNumberOfPulses(uint32_t& numberOfPulses);
   int SetStart(uint32_t& response);
   int SetStop(uint32_t& response);
   int SetInterval(uint32_t interval, uint32_t& response);
   int SetPulseDuration(uint32_t pulseDuration, uint32_t& response);
   int SetWaitForInput(uint32_t waitForInput, uint32_t& response);
   int SetNumberOfPulses(uint32_t numberOfPulses, uint32_t& response);


private:
   MM::Core* callback_;
   MM::Device* device_;
   const char* portLabel_;
   std::mutex mutex_;
};
