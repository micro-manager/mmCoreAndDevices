#ifndef __prizmatix_main_
#define __prizmatix_main_
#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
 
#define ERR_INITIALIZE_FAILED 102
#define ERR_WRITE_FAILED 103
#define ERR_CLOSE_FAILED 104
#define ERR_BOARD_NOT_FOUND 105
#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109

 
 
class PrizmatixHub : public HubBase<PrizmatixHub>  
{
public:
   PrizmatixHub();
   ~PrizmatixHub();
   int GetNmLEDS(){return nmLeds;};
   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();

   // property handlers
   int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
  //MMM???? int OnLogic(MM::PropertyBase* pPropt, MM::ActionType eAct);
   int OnVersion(MM::PropertyBase* pPropt, MM::ActionType eAct);

   // custom interface for child devices
   bool IsPortAvailable() {return portAvailable_;}
   bool IsLogicInverted() {return invertedLogic_;}
   bool IsTimedOutputActive() {return timedOutputActive_;}
   void SetTimedOutput(bool active) {timedOutputActive_ = active;}

   int PurgeComPortH() {return PurgeComPort(port_.c_str());}
   int WriteToComPortH(const unsigned char* command, unsigned len) {return WriteToComPort(port_.c_str(), command, len);}
   int ReadFromComPortH(unsigned char* answer, unsigned maxLen, unsigned long& bytesRead)
   {
      return ReadFromComPort(port_.c_str(), answer, maxLen, bytesRead);
   }
     int  SendSerialCommandH(char *b)
	 {
		 if(initialized_==false) return 0;
		 return SendSerialCommand(port_.c_str(), b,"\n");
	 }
	  int  GetSerialAnswerH( std::string &answer)
	 {
		   
    return  GetSerialAnswer(port_.c_str(), "\r\n", answer);
		  
	 }
	  static MMThreadLock lock_;
   static MMThreadLock& GetLock() {return lock_;}
  
   void SetShutterState(unsigned state) {shutterState_ = state;}
   void SetSwitchState(unsigned state) {switchState_ = state;}
   unsigned GetShutterState() {return shutterState_;}
   unsigned GetSwitchState() {return switchState_;}
   int GetNmLeds() {return nmLeds;}
private:
	int nmLeds;
	
   int GetControllerVersion(int&);
   std::string port_;
   bool initialized_;
   bool portAvailable_;
   bool invertedLogic_;
   bool timedOutputActive_;
   int version_;
   
   unsigned switchState_;
   unsigned shutterState_;
};
//CGenericBase
 
//PrizmatixLED   CSignalIOBase
class PrizmatixLED : public CGenericBase<PrizmatixLED>  
{
public:
   PrizmatixLED(int nmLeds,char *Name);
   ~PrizmatixLED();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   

   // DA API
      virtual bool Busy()
	  {
		  return false;
	  }
	   /*  int SetGateOpen(bool open) {return DEVICE_OK;};  // abstract function in paret
	int GetGateOpen(bool& open) { return DEVICE_OK;};
	int SetSignal(double volts){return DEVICE_OK;} ;
	int GetSignal(double& volts) { return DEVICE_UNSUPPORTED_COMMAND;}     
	int GetLimits(double& minVolts, double& maxVolts) {return DEVICE_OK;}
   
   int IsDASequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}
   */
   // action interface
   // ----------------
 
   int OnPowerLEDEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param);
   int OnOfOnEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param);
   int OnSTBL(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	PrizmatixHub* myHub;
   int WriteToPort(char * Str);
   long  ValLeds[10];
   long  OnOffLeds[10];
   bool initialized_;
   int nmLeds;
   std::string name_;
};




 
 
#endif