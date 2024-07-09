#ifndef _SQUID_H_
#define _SQUID_H_

#include "MMDevice.h"
#include "DeviceBase.h"


#define ERR_PORT_CHANGE_FORBIDDEN    21001 

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

   static const int RCV_BUF_LENGTH = 1024;
   unsigned char rcvBuf_[RCV_BUF_LENGTH];
   std::string port_;

private:
   uint8_t crc8ccitt(const void* data, size_t size);
   int sendCommand(unsigned char* cmd, unsigned cmdSize);
   bool initialized_;
   SquidMonitoringThread* monitoringThread_;
};


/*
class SquidShutter : public CShutterBase<SquidShutter>
{
   public:
      SquidShutter();
      ~SquidShutter();

      int Initialize();
      int Shutdown();

      void GetName(char* pszName) const;
      bool Busy();
      unsigned long GetShutterNr() const { return shutterNr_; }

      // Shutter API
      int SetOpen(bool open = true);
      int GetOpen(bool& open);
      int Fire(double deltaT);

      // action interface
      int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnShutterNr(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnExternal(MM::PropertyBase* pProp, MM::ActionType eAct);

   private:
      bool initialized_;
      std::string name_;
      unsigned shutterNr_;
      bool external_;
      bool state_;
      MM::MMTime changedTime_;
}
*/

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