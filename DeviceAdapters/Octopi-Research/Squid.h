#ifndef _SQUID_H_
#define _SQUID_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <cstdint>
#include <thread>
#include <mutex>


#define ERR_PORT_CHANGE_FORBIDDEN    21001 
#define ERR_NO_PORT_SET 21002

extern const char* g_HubDeviceName;
extern const char* g_LEDShutterName;
extern const char* g_XYStageName;

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

const int ILLUMINATION_SOURCE_LED_ARRAY_FULL = 0;
const int ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF = 1;
const int ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF = 2;
const int ILLUMINATION_SOURCE_LED_ARRAY_LEFTB_RIGHTR = 3;
const int ILLUMINATION_SOURCE_LED_ARRAY_LOW_NA = 4;
const int ILLUMINATION_SOURCE_LED_ARRAY_LEFT_DOT = 5;
const int ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_DOT = 6;


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
   int SendCommand(unsigned char* cmd, unsigned cmdSize, uint8_t* cmdNr);
   int SendMoveCommand(const int cmd, long steps);
   bool IsCommandPending(uint8_t cmdNr);
   void ReceivedCommand(uint8_t cmdNr);
   int GetPositionSteps(long& x, long& y);
   int SetPositionXSteps(long x);
   int SetPositionYSteps(long y);
   int SetPositionZSteps(long z);

   std::string port_;

private:
   void SetCommandPending(uint8_t cmdNr);
   bool initialized_;
   SquidMonitoringThread* monitoringThread_;
   uint8_t cmdNr_;
   uint8_t pendingCmd_;
   std::recursive_mutex lock_;
   std::recursive_mutex positionLock_;
   std::atomic_long x_;
   std::atomic_long y_;
   std::atomic_long z_;
};



class SquidLEDShutter : public CShutterBase<SquidLEDShutter>
{
public:
   SquidLEDShutter();
   ~SquidLEDShutter();

   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Shutter API
   int SetOpen(bool open = true);
   int GetOpen(bool& open);
   int Fire(double deltaT);

   // action interface
   int OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRed(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGreen(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBlue(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
   int sendIllumination(uint8_t pattern, uint8_t intensity, uint8_t red, uint8_t green, uint8_t blue);
   SquidHub* hub_;
   bool initialized_;
   std::string name_;
   MM::MMTime changedTime_;
   uint8_t pattern_;
   uint8_t intensity_;
   uint8_t red_;
   uint8_t green_;
   uint8_t blue_;
   bool isOpen_;
   uint8_t cmdNr_;
};


class SquidXYStage : public CXYStageBase<SquidXYStage>
{
public:
   SquidXYStage();
   ~SquidXYStage();
   int Shutdown();
   void GetName(char* pszName) const;

   int Initialize();

   bool Busy();

   //double GetStepSize() { return stepSize_um_; }
   int SetPositionSteps(long x, long y);
   int GetPositionSteps(long& x, long& y)
   {
      return hub_->GetPositionSteps(x, y);
   }
   int SetRelativePositionSteps(long x, long y);
   int Home()
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   int Stop()
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   /* This sets the 0,0 position of the adapter to the current position.
    * If possible, the stage controller itself should also be set to 0,0
    * Note that this differs form the function SetAdapterOrigin(), which
    * sets the coordinate system used by the adapter
    * to values different from the system used by the stage controller
    */
   int SetOrigin() { return DEVICE_OK; }

   int GetLimitsUm(double& /*xMin*/, double& /* xMax */, double& /* yMin */, double& /* yMax */)
   {
      //xMin = lowerLimit_; xMax = upperLimit_;
      //yMin = lowerLimit_; yMax = upperLimit_;
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   int GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   double GetStepSizeXUm() { return stepSizeX_um_; }
   double GetStepSizeYUm() { return stepSizeY_um_; }
   int Move(double /*vx*/, double /*vy*/) { return DEVICE_OK; }

   int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }


   // action interface
   // ----------------
   int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnModel(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
   SquidHub* hub_;
   double stepSizeX_um_;
   double stepSizeY_um_;
   double screwPitchXmm_;
   double screwPitchYmm_;
   double fullStepsPerRevX_;
   double fullStepsPerRevY_;
   int microSteppingDefaultX_;  // needs to be set as pre-init tied to model
   int microSteppingDefaultY_;  // needs to be set as pre-init tied to model
   double posX_um_;
   double posY_um_;
   bool busy_;
   MM::TimeoutMs* timeOutTimer_;
   double velocity_;
   bool initialized_;
   uint8_t cmdNr_;

};


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


class SquidMonitoringThread {
public:
   SquidMonitoringThread(MM::Core& core, SquidHub& hub, bool debug);
   ~SquidMonitoringThread();
   int svc();

   void Start();
   void Stop() { stop_ = true; }

private:
   void InterpretMessage(unsigned char* message);
   bool IsBigEndian(void);
   static const int RCV_BUF_LENGTH = 1024;
   MM::Core& core_;
   SquidHub& hub_;
   bool debug_;
   bool stop_;
   long intervalUs_;
   std::thread* ourThread_;
   bool isBigEndian_;
   SquidMonitoringThread& operator=(SquidMonitoringThread& /*rhs*/) { assert(false); return *this; }
};

#endif _SQUID_H_