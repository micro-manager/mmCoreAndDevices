 //////////////////////////////////////////////////////////////////////////////
// FILE:          MicroFPGA.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for MicroFPGA, a FPGA platform using FPGA boards from
//                Alchitry. The adapter must be used with a special firmware, see:
//                https://github.com/jdeschamps/MicroFPGA
// COPYRIGHT:     EMBL
// LICENSE:       LGPL
//
// AUTHOR:        Joran Deschamps, EMBL, 2020
//
//

#ifndef _Mojo_H_
#define _Mojo_H_

#include "MMDevice.h"
#include "DeviceBase.h"

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_BOARD_NOT_FOUND 101
#define ERR_PORT_OPEN_FAILED 102
#define ERR_NO_PORT_SET 103
#define ERR_VERSION_MISMATCH 104
#define ERR_UNKNOWN_ID 105
#define ERR_COMMAND_UNKNOWN 11206655


class MicroFPGAHub : public HubBase<MicroFPGAHub>  
{
public:
   MicroFPGAHub();
   ~MicroFPGAHub();

   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();

   // property handlers
   int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);

   int PurgeComPortH() {return PurgeComPort(port_.c_str());}
   int SendWriteRequest(long address, long value);
   int SendReadRequest(long address);
   int ReadAnswer(long& answer);
   int WriteToComPortH(const unsigned char* command, unsigned len) {return WriteToComPort(port_.c_str(), command, len);}
   int ReadFromComPortH(unsigned char* answer, unsigned maxLen, unsigned long& bytesRead) {
      return ReadFromComPort(port_.c_str(), answer, maxLen, bytesRead);
   }
   static MMThreadLock& GetLock() {return lock_;}

private:
   int GetControllerVersion(long&);
   int GetID(long&);
   std::string port_;
   bool initialized_;
   bool portAvailable_;
   long version_;
   long id_;
   static MMThreadLock lock_;
};


///////////////////////////////////////////////////////////////////////////////////////////
//////
class LaserTrig   : public CGenericBase<LaserTrig>  
{
public:
   LaserTrig();
   ~LaserTrig();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy() {return busy_;}
   
   unsigned long GetNumberOfLasers()const {return numlasers_;}

   // action interface
   // ----------------
   int OnMode(MM::PropertyBase* pProp, MM::ActionType eAct, long laser);
   int OnDuration(MM::PropertyBase* pProp, MM::ActionType eAct, long laser);
   int OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct, long laser);
   int OnNumberOfLasers(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	
   int WriteToPort(long address, long value);
   int ReadFromPort(long& answer);

   bool initialized_;
   long numlasers_;
   long *mode_;
   long *duration_;
   long *sequence_;
   bool busy_;
};

///////////////////////////////////////////////////////////////////////////////////////////
//////

class Servo : public CGenericBase<Servo>  
{
public:
   Servo();
   ~Servo();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy() {return busy_;}
   
   unsigned long GetNumberOfServos()const {return numServos_;}

   // action interface
   // ----------------
   int OnNumberOfServos(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct, long servo);

private:
   int WriteToPort(long address, long value);
   int ReadFromPort(long& answer);

   long *position_;
   bool initialized_;
   long numServos_;
   bool busy_;
};

///////////////////////////////////////////////////////////////////////////////////////////
//////

class TTL : public CGenericBase<TTL>  
{
public:
   TTL();
   ~TTL();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy() {return busy_;}
   
   unsigned long GetNumberOfChannels()const {return numChannels_;}

   // action interface
   // ----------------
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct, long channel);
   int OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int WriteToPort(long channel, long state);
   int ReadFromPort(long& answer);

   long numChannels_;
   long *state_;
   bool initialized_;
   bool busy_;
};

///////////////////////////////////////////////////////////////////////////////////////////
//////

class PWM : public CGenericBase<PWM>  
{
public:
   PWM();
   ~PWM();
  
   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy() {return busy_;}
   
   unsigned long GetNumberOfChannels()const {return numChannels_;}

   // action interface
   // ----------------
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct, long servo);
   int OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int WriteToPort(long channel, long value);
   int ReadFromPort(long& answer);
   
   bool initialized_;
   long *state_;
   long numChannels_;
   bool busy_;
};


///////////////////////////////////////////////////////////////////////////////////////////
//////
class AnalogInput : public CGenericBase<AnalogInput>  
{
public:
   AnalogInput();
   ~AnalogInput();

   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   int OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType eAct, long channel);
   int OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType eAct);
   
   unsigned long GetNumberOfChannels()const {return numChannels_;}

private:
   int WriteToPort(long address);
   int ReadFromPort(long& answer);
   
   MMThreadLock lock_;
   long numChannels_;
   long *state_;
   bool initialized_;
};

#endif