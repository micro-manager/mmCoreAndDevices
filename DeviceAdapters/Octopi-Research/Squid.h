#ifndef _SQUID_H_
#define _SQUID_H_

#include "MMDevice.h"
#include "DeviceBase.h"


#define ERR_PORT_CHANGE_FORBIDDEN    21001 
#define ERR_NO_PORT_SET 21002


const char* g_HubDeviceName = "SquidHub";
const char* g_ShutterName = "LEDs";

const int CMD_MOVE_X = 0;
const int CMD_MOVE_Y = 1;
const int CMD_MOVE_Z = 2;
const int CMD_MOVE_THETA = 3;
const int CMD_HOME_OR_ZERO = 5;
const int CMD_TURN_ON_ILLUMINATION = 10;
const int CMD_TURN_OFF_ILLUMINATION = 11;
const int CMD_SET_ILLUMINATION = 12;
const int CMD_SET_ILLUMINATION_LED_MATRIX = 13;
const int CMD_ACK_JOYSTICK_BUTTON_PRESSED = 14;
const int CMD_ANALOG_WRITE_ONBOARD_DAC = 15;
const int CMD_SET_DAC80508_REFDIV_GAIN = 16;
const int CMD_SET_ILLUMINATION_INTENSITY_FACTOR = 17;
const int CMD_MOVETO_X = 6;
const int CMD_MOVETO_Y = 7;
const int CMD_MOVETO_Z = 8;
const int CMD_SET_LIM = 9;
const int CMD_SET_LIM_SWITCH_POLARITY = 20;
const int CMD_CONFIGURE_STEPPER_DRIVER = 21;
const int CMD_SET_MAX_VELOCITY_ACCELERATION = 22;
const int CMD_SET_LEAD_SCREW_PITCH = 23;
const int CMD_SET_OFFSET_VELOCITY = 24;
const int CMD_CONFIGURE_STAGE_PID = 25;
const int CMD_ENABLE_STAGE_PID = 26;
const int CMD_DISABLE_STAGE_PID = 27;
const int CMD_SET_HOME_SAFETY_MERGIN = 28;
const int CMD_SET_PID_ARGUMENTS = 29;
const int CMD_SEND_HARDWARE_TRIGGER = 30;
const int CMD_SET_STROBE_DELAY = 31;
const int CMD_SET_PIN_LEVEL = 41;
const int CMD_INITIALIZE = 254;
const int CMD_RESET = 255;


class SquidMonitoringThread;

class SquidHub : public HubBase<SquidHub>
{
public:
   SquidHub();
   ~SquidHub();

   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();

   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

   bool IsPortAvailable() { return (port_ != ""); };
   int SendCommand(unsigned char* cmd, unsigned cmdSize);

   static const int RCV_BUF_LENGTH = 1024;
   unsigned char rcvBuf_[RCV_BUF_LENGTH];
   std::string port_;

private:
   uint8_t crc8ccitt(const void* data, size_t size);
   bool initialized_;
   SquidMonitoringThread* monitoringThread_;
};



class SquidShutter : public CShutterBase<SquidShutter>
{
   public:
      SquidShutter();
      ~SquidShutter();

      int Initialize();
      int Shutdown();

      void GetName(char* pszName) const;
      bool Busy();

      // Shutter API
      int SetOpen(bool open = true);
      int GetOpen(bool& open);
      int Fire(double deltaT);

      // action interface
      int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);

   private:
      bool initialized_;
      std::string name_;
      unsigned shutterNr_;
      bool external_;
      bool state_;
      MM::MMTime changedTime_;
}


class SquidMessageParser {
public:
   SquidMessageParser(unsigned char* inputStream, long inputStreamLength);
   ~SquidMessageParser() {};
   int GetNextMessage(unsigned char* nextMessage, int& nextMessageLength);
   static const int messageMaxLength_ = 24;

private:
   unsigned char* inputStream_;
   long inputStreamLength_;
   long index_;
};


class SquidMonitoringThread : public MMDeviceThreadBase
{
public:
   SquidMonitoringThread(MM::Core& core, SquidHub& hub, bool debug);
   ~SquidMonitoringThread();
   int svc();
   int open(void*) { return 0; }
   int close(unsigned long) { return 0; }

   void Start();
   void Stop() { stop_ = true; }

private:
   void interpretMessage(unsigned char* message);
   //MM::Device& device_;
   MM::Core& core_;
   SquidHub& hub_;
   bool debug_;
   bool stop_;
   long intervalUs_;
   SquidMonitoringThread& operator=(SquidMonitoringThread& /*rhs*/) { assert(false); return *this; }
};

#endif _SQUID_H_