#ifndef _SQUID_H_
#define _SQUID_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>


#define ERR_PORT_CHANGE_FORBIDDEN    21001 
#define ERR_NO_PORT_SET 21002

extern const char* g_HubDeviceName;
extern const char* g_ShutterName;
extern const char* g_XYStageName;
extern const char* g_ZStageName;
extern const char* g_DAName;

const unsigned char CMD_MOVE_X = 0;
const unsigned char CMD_MOVE_Y = 1;
const unsigned char CMD_MOVE_Z = 2;
const unsigned char CMD_MOVE_THETA = 3;
const unsigned char CMD_HOME_OR_ZERO = 5;
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

const unsigned char AXIS_X = 0;
const unsigned char AXIS_Y = 1;
const unsigned char AXIS_Z = 2;
const unsigned char AXIS_THETA = 3;
const unsigned char AXIS_XY = 4;

const int STAGE_MOVEMENT_SIGN_X = -1;
const int STAGE_MOVEMENT_SIGN_Y = -1;
const int STAGE_MOVEMENT_SIGN_Z = -1;

const unsigned char COMPLETED_WITHOUT_ERRORS = 0;
const unsigned char IN_PROGRESS = 1;
const unsigned char CMD_CHECKSUM_ERROR = 2;
const unsigned char CMD_INVALID = 3;
const unsigned char CMD_EXECUTION_ERROR = 4;
const unsigned char ERROR_CODE_EMPTYING_THE_FLUDIIC_LINE_FAILED = 10;

extern const char* g_AutoHome;
extern const char* g_Yes;
extern const char* g_No;
extern const char* g_Acceleration;
extern const char* g_Max_Velocity;

class SquidMonitoringThread;
class SquidXYStage;
class SquidZStage;

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
   int OnAutoHome(MM::PropertyBase* pProp, MM::ActionType eAct);

   bool IsPortAvailable() { return (port_ != ""); };
   int SendCommand(unsigned char* cmd, unsigned cmdSize);
   int SendMoveCommand(const int cmd, long steps);
   int SetMaxVelocityAndAcceleration(unsigned char axis, double maxVelocity, double acceleration);
   int Home();
   void GetPositionXYSteps(long& x, long& y);
   void GetPositionZSteps(long& z);
   void SetPositionXSteps(long x);
   void SetPositionYSteps(long y);
   void SetPositionZSteps(long z);
   void SetCmdNrReceived(uint8_t cmdNr, uint8_t status);

   std::string port_;

   int AssignXYStageDevice(SquidXYStage* xyStageDevice);
   int AssignZStageDevice(SquidZStage* zStageDevice);

   bool XYStageBusy();
   bool ZStageBusy();

   int SetDacGain(uint8_t dacNr, bool gain);


private:
   bool initialized_;
   bool autoHome_;
   SquidMonitoringThread* monitoringThread_;
   SquidXYStage* xyStageDevice_;
   SquidZStage* zStageDevice_;
   std::atomic_long x_;
   std::atomic_long y_;
   std::atomic_long z_;
   std::atomic<unsigned char> cmdNrSend_;
   std::atomic<unsigned char> cmdNrReceived_;
   std::atomic<unsigned char> status_;
   std::atomic_bool xStageBusy_;
   std::atomic_bool yStageBusy_;
   std::atomic_bool zStageBusy_;
   std::atomic_bool busy_;
   std::atomic<unsigned char> dac_div_; // 0: 1x gain: 2.5V, 1: 1x gain: 1.25V
   std::atomic<unsigned char> dac_gains_; // bit mask with gains.  0 multiples dac_div with 1, 1 multiplies dac_div with 2
               // i.e. dac_div_ 1 and dac_gains_ 1 for a given output results in 0-5V range.
   std::mutex mutex_;
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
   int OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnPattern(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRed(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGreen(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBlue(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnHasLasers(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserIntensity(MM::PropertyBase* pProp, MM::ActionType eAct, long laserLine);


private:
   int sendIllumination(uint8_t pattern, uint8_t intensity, uint8_t red, uint8_t green, uint8_t blue);
   int sendLaserIllumination(uint8_t pattern, uint16_t intensity);
   SquidHub* hub_;
   bool initialized_;
   bool hasLasers_;
   std::string name_;
   MM::MMTime changedTime_;
   uint8_t pattern_;
   uint8_t intensity_;
   uint8_t red_;
   uint8_t green_;
   uint8_t blue_;
   uint16_t iLaser_[5];
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

   int SetPositionSteps(long x, long y);
   int GetPositionSteps(long& x, long& y)
   {
      hub_->GetPositionXYSteps(x, y);
      return DEVICE_OK;
   }
   int SetRelativePositionSteps(long x, long y);
   int Home();
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
   int SetOrigin() { return DEVICE_UNSUPPORTED_COMMAND; }

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
   int Move(double /*vx*/, double /*vy*/) { return DEVICE_UNSUPPORTED_COMMAND; }

   int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }


   // action interface
   // ----------------
   int OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);

   int Callback(long xSteps, long ySteps);

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
   double maxVelocity_;
   double acceleration_;
   bool initialized_;
   uint8_t cmdNr_;

};


class SquidZStage : public CStageBase<SquidZStage>
{
public:
   SquidZStage();
   ~SquidZStage();
   int Shutdown();
   void GetName(char* pszName) const;
   int Initialize();

   bool Busy();

   double GetStepSize() { return stepSize_um_; }
   int SetPositionUm(double pos);
   int GetPositionUm(double& pos);
   int SetRelativePositionUm(double d);
   int SetPositionSteps(long z);
   int GetPositionSteps(long& z);
   int SetRelativePositionSteps(long z);
   int Home();
   int Stop();

   int SetOrigin()
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   };
   int SetOrigin(double&)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   };
   int GetLimits(double&, double&)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   };
   int IsStageSequenceable(bool& sequenceable) const {
      sequenceable = false;
      return DEVICE_OK;
   }
   bool IsContinuousFocusDrive() const { return false; };

   int OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);

   int Callback(long zSteps);

private:
   SquidHub* hub_;
   double stepSize_um_;
   double screwPitchZmm_;
   double microSteppingDefaultZ_; 
   double fullStepsPerRevZ_;
   double maxVelocity_;
   double acceleration_;
   bool initialized_;
   uint8_t cmdNr_;
};


class SquidDA : public CSignalIOBase<SquidDA>
{
public:
   SquidDA(uint8_t dacNr);
   ~SquidDA();

   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // DA API
   int SetGateOpen(bool open);
   int GetGateOpen(bool& open);
   int SetSignal(double volts);
   int GetSignal(double& volts);
   int GetLimits(double& minVolts, double& maxVolts);

   int IsDASequenceable(bool& isSequenceable) const;

   // action interface
   // ----------------
   int OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVoltRange(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int SendVoltage(double volt);
   SquidHub* hub_;

   bool initialized_;
   bool busy_;
   double maxV_;
   double volts_;
   double gatedVolts_;
   long dacNr_;
   bool gateOpen_;
   std::string name_;
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
   static const int RCV_BUF_LENGTH = 48; // contains 2 messages.  If set to 24, we crash...
   MM::Core& core_;
   SquidHub& hub_;
   bool debug_;
   bool stop_;
   long intervalUs_;
   std::thread* ourThread_;
   bool isBigEndian_;
   unsigned int counter_;
   SquidMonitoringThread& operator=(SquidMonitoringThread& /*rhs*/) { assert(false); return *this; }
};

#endif
