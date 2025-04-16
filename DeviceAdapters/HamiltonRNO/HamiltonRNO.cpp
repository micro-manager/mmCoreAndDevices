/*
 * Micro-Manager device adapter for Hamilton devices that use the RNO protocol
 *
 * Author: Mark A. Tsuchida <mark@open-imaging.com> for the original MVP code
 *         Egor Zindy <ezindy@gmail.com> for the PSD additions
 *
 * Copyright (C) 2018 Applied Materials, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "RNOCommands.h"

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ModuleInterface.h"

#include <string>
#include <vector>


//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107

const char* const DEVICE_NAME_HUB = "HamiltonChain";
const char* const DEVICE_NAME_MVP_PREFIX = "HamiltonMVP-";
const char* const DEVICE_NAME_PSD_PREFIX = "HamiltonPSD-";
const char* const DEVICE_NAME_MVP_FULLNAME = "Modular Valve Positioner";
const char* const DEVICE_NAME_PSD_FULLNAME = "Precision Syringe Drive";

const char* const g_Keyword_BackOffSteps = "BackOffSteps";
const char* const g_Keyword_ReturnSteps = "ReturnSteps";
const char* const g_Keyword_FullStrokeSteps = "FullStrokeSteps";
const char* const g_Keyword_HalfResolution = "Half Resolution";
const char* const g_Keyword_FullResolution = "Full Resolution";
const char* const g_Keyword_FullResolutionDisabled = "Full Res No Overload Detection";

enum {
   ERR_UNKNOWN_VALVE_TYPE = 21001,
   ERR_INITIALIZATION_TIMED_OUT,
};


// Treat a chain of MVPs on the same serial port as a "hub"
class MVPChain : public HubBase<MVPChain>
{
   std::string port_;
   char maxAddr_;

public:
   MVPChain();
   virtual ~MVPChain();

   virtual void GetName(char* name) const;
   virtual int Initialize();
   virtual int Shutdown();
   virtual bool Busy();

   virtual int DetectInstalledDevices();

private: // Property handlers
   int OnPort(MM::PropertyBase *pProp, MM::ActionType eAct);

public: // For access by peripherals
   int SendRecv(HamiltonCommand& cmd);
   int SelectChannel(char address, int index);
};


class MVP : public CStateDeviceBase<MVP>
{
   const char address_;
   const std::string firmware_;
   int index_channel_;
   int index_E2_;
   DeviceType deviceType_;

   MVPValveType valveType_;

   enum RotationDirection {
      CLOCKWISE,
      COUNTERCLOCKWISE,
      LEAST_ANGLE,
   } rotationDirection_;

   MMThreadLock lock_;


public:
   MVP(char address, int index);
   virtual ~MVP();

   virtual void GetName(char* name) const;
   virtual int Initialize();
   virtual int Shutdown();
   virtual bool Busy();

   virtual int GetPosition(long& pos);
   virtual int SetPosition(long pos);

   virtual unsigned long GetNumberOfPositions() const;

private: // Property handlers
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRotationDirection(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int SendRecv(HamiltonCommand& cmd);
   int SelectChannel();

   bool ShouldRotateCCW(int curPos, int newPos);
   static std::string RotationDirectionToString(RotationDirection rd);
   static RotationDirection RotationDirectionFromString(const std::string& s);

private:
   MVP& operator=(const MVP&); // = delete
};


class PSD : public CVolumetricPumpBase<PSD>
{
   const char address_;
   const std::string firmware_;
   int index_channel_;
   int index_E2_;

   double flowrate_ul_per_sec_;
   int speed_;
   long back_off_steps_;
   long return_steps_;
   DeviceType deviceType_;

   // full stroke is device dependant and cannot be queried.
   long full_stroke_steps_;

   // current Volume
   double volume_ul_;
   // max Volume
   double volumeMax_ul_;

   bool busy_;
   bool initialized_;

   enum DriveResolution {
      HALFRESOLUTION=0,
      FULLRESOLUTION = 1,
      FULLRESOLUTION_OVERLOADDISABLED = 2,
   } driveResolution_;

   MMThreadLock lock_;

public:
   PSD(char address, int index);
   virtual ~PSD();

   virtual void GetName(char* name) const;
   virtual int Initialize();
   virtual int Shutdown();
   virtual bool Busy();

   /*
   virtual int Aspirate();
   virtual int AspirateVolumeUl(double volUl);
   virtual int Dispense();
   virtual int DispenseVolumeUl(double volUl);
   */
   virtual int Home();
   virtual int Stop();
   //virtual bool RequiresHoming();
   //virtual int InvertDirection(bool state);
   //virtual int IsDirectionInverted(bool& state);
   //virtual int SetVolumeUl(double volUl);
   //virtual int GetVolumeUl(double& volUl);
   //virtual int SetMaxVolumeUl(double volUl);
   //virtual int GetMaxVolumeUl(double& volUl);
   //virtual int SetFlowrateUlPerSecond(double flowrate);
   //virtual int GetFlowrateUlPerSecond(double& flowrate);
   virtual int Start();
   virtual int DispenseDurationSeconds(double durSec);
   virtual int DispenseVolumeUl(double volUl);

   bool RequiresHoming()
   {
      return false;
   }
   int InvertDirection(bool state)
   {
      return DEVICE_OK;
   }
   int IsDirectionInverted(bool& state)
   {
      return false;
   }

   int SetFlowrateUlPerSecond(double flowrate)
   {
      flowrate_ul_per_sec_ = flowrate;
      speed_ = (int)(volumeMax_ul_ / flowrate_ul_per_sec_);
      return DEVICE_OK;
   }

   int GetFlowrateUlPerSecond(double& flowrate)
   {
      flowrate = flowrate_ul_per_sec_ ;
      LogMessage("Reporting flowrate (ul/sec)", true);
      return DEVICE_OK;
   }

   int SetVolumeUl(double volUl)
   {
      volume_ul_ = volUl;
      return DEVICE_OK;
   }

   int GetVolumeUl(double& volUl)
   {
      int err; 
      MMThreadGuard myLock(lock_);
      {
         err = SelectChannel();
         SyringePositionRequest spReq(address_);

         if (err == DEVICE_OK)
            err = SendRecv(spReq);

         if (err == DEVICE_OK)
            volume_ul_ = spReq.GetPosition() * volumeMax_ul_ / full_stroke_steps_;
      }
      if (err == DEVICE_OK)
         volUl = volume_ul_;

      return err;
   }

   int SetMaxVolumeUl(double volUl)
   {
      volumeMax_ul_ = volUl;
      speed_ = (int)(volumeMax_ul_ / flowrate_ul_per_sec_);

      //get and set volume_ul_
      int err = GetVolumeUl(volUl);
      return err;
   }

   int GetMaxVolumeUl(double& volUl)
   {
      volUl = volumeMax_ul_;
      return DEVICE_OK;
   }

private:
   int SetBackOffSteps(int steps);
   int SetReturnSteps(int steps);
   int SetDriveResolution(int resolution);
   int SendRecv(HamiltonCommand& cmd);
   int SelectChannel();
   static std::string DriveResolutionToString(DriveResolution rd);
   static DriveResolution DriveResolutionFromString(const std::string& s);

private: // Property handlers
   int OnDriveResolution(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFullStrokeSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFlowrate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVolume(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxVolume(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReturnSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBackOffSteps(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   PSD& operator=(const PSD&); // = delete
};


MODULE_API void InitializeModuleData()
{
   RegisterDevice(DEVICE_NAME_HUB, MM::HubDevice,
         "Hamilton Modular Valve Positioner (possibly chained)");
}


MODULE_API MM::Device* CreateDevice(const char* name)
{
   //The pump or valve index (to account for multi-valve / multi-pump systems
   int index;
   char address;
   char* description;
   MM::Device* device;

   if (!name)
      return 0;

   // Create a Hub device
   if (strcmp(name, DEVICE_NAME_HUB) == 0)
      return new MVPChain();

   // Create a MVP device:
   //
   // NOTE: Some may have multiple valves and pumps per device,
   // hence the need for a channel index.
   // The index is used to determine the position of the E2 information byte.
   //
   // Index values can be: 0 (one device, no index selection), 1 (left) or 2 (right).
   // The MVP or PSD would use 0 as the index, the Microlab uses 1 and 2, so we know we
   // need to select left or right devices before issuing a command.
   //
   // The name generated also depends on the index value:
   // * With 0, just do a prefix-device
   // * With 1 or mode, do prefix-deviceIndex where device is [a-z] and index is [1-4]
   //
   if (strncmp(name, DEVICE_NAME_MVP_PREFIX, strlen(DEVICE_NAME_MVP_PREFIX)) == 0)
   {
      // Address must be 1 or 2 char
      if (strlen(name) - strlen(DEVICE_NAME_MVP_PREFIX) > 2 || strlen(name) - strlen(DEVICE_NAME_MVP_PREFIX) == 0)
         return 0;

      if (strlen(name) - strlen(DEVICE_NAME_MVP_PREFIX) == 2)
      {
         address = name[strlen(name) - 2];
         index = (int)(name[strlen(name) - 1]) - '0';
      }
      else
      {
         address = name[strlen(name) - 1];
         index = 0;
      }

      device = new MVP(address, index);
      RegisterDevice(name, MM::StateDevice, DEVICE_NAME_MVP_FULLNAME);
      return device;
   }

   // Create a PSD device
   if (strncmp(name, DEVICE_NAME_PSD_PREFIX, strlen(DEVICE_NAME_PSD_PREFIX)) == 0)
   {
      // Address must be 1 or 2 char
      if (strlen(name) - strlen(DEVICE_NAME_PSD_PREFIX) > 2 || strlen(name) - strlen(DEVICE_NAME_PSD_PREFIX) == 0)
         return 0;

      if (strlen(name) - strlen(DEVICE_NAME_PSD_PREFIX) == 2)
      {
         address = name[strlen(name) - 2];
         index = (int)(name[strlen(name) - 1]) - '0';
      }
      else
      {
         address = name[strlen(name) - 1];
         index = 0;
      }

      device = new PSD(address, index);
      RegisterDevice(name, MM::VolumetricPumpDevice, DEVICE_NAME_PSD_FULLNAME);
      return device;
   }

   return 0;
}


MODULE_API void DeleteDevice(MM::Device* device)
{
   delete device;
}


///////////////////////////////////////////////////////////////////////////////
// Hamilton Hub
///////////////////////////////////////////////////////////////////////////////

MVPChain::MVPChain() :
   port_("Undefined"),
   maxAddr_('a')
{
   CreateStringProperty("Port", port_.c_str(), false,
         new CPropertyAction(this, &MVPChain::OnPort), true);
}


MVPChain::~MVPChain()
{
}


void
MVPChain::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, DEVICE_NAME_HUB);
}


int
MVPChain::Initialize()
{
   int err;

   //Send a global reset, no answer to be expected.
   err = SendSerialCommand(port_.c_str(), ":!", RNO_TERM);
   if (err != DEVICE_OK)
      return err;

   //Try 5 times
   for (int i=0; i < 5; i++)
   { 

      CDeviceUtils::SleepMs(500);
      std::string answer;
      err = GetSerialAnswer(port_.c_str(), RNO_TERM, answer);
      if (err == DEVICE_OK)
         break; 
   } 

   AutoAddressingCommand autoaddr;
   err = SendRecv(autoaddr);
   if (err != DEVICE_OK)
      return err;

   if (autoaddr.HasMaxAddr())
   {
      maxAddr_ = autoaddr.GetMaxAddr();
   }
   else
   {
      // Autoaddressing did not happen, presumably because the MVPs have
      // already been assigned addresses.
      // In this case, we test each address.
      char addr;
      for (addr = 'a'; addr < 'z'; ++addr)
      {
         FirmwareVersionRequest req(addr);
         err = SendRecv(req);
         if (err)
         {
            if (addr == 'a')
               return err;
            break;
         }
      }
      maxAddr_ = addr - 1;
   }

   LogMessage(("Last address in chain is '" + std::string(1, maxAddr_) + "'").c_str());

   return DEVICE_OK;
}


int
MVPChain::Shutdown()
{
   return DEVICE_OK;
}


bool
MVPChain::Busy()
{
   return false;
}


int
MVPChain::DetectInstalledDevices()
{
   ClearInstalledDevices();

   int err = DEVICE_OK;
   std::string firmware;
   std::string deviceName;

   for (char addr = 'a'; addr <= maxAddr_; ++addr)
   {
      MM::Device* device;

      FirmwareVersionRequest fvReq(addr);
      err = SendRecv(fvReq);
      if (err != DEVICE_OK)
         break;

      firmware = fvReq.GetFirmwareVersion();
      DeviceType deviceType = GetDeviceTypeFromFirmware(firmware);
      deviceName = GetDeviceTypeName(deviceType);
      LogMessage(("Detecting device '" + std::string(1, addr) + "' " + firmware + " " + deviceName).c_str(), true);

      switch (deviceType)
      {
         case DeviceTypeML600: case DeviceTypeML500A: case DeviceTypeML500B: case DeviceTypeML500C: case DeviceTypeML700:
            // FIXME need to check these against actual hardware, I'm just assuming here they are dual pump / dual valve systems 
            device = new MVP(addr, 1);
            if (device)
               AddInstalledDevice(device);
            device = new MVP(addr, 2);
            if (device)
               AddInstalledDevice(device);
            device = new PSD(addr, 1);
            if (device)
               AddInstalledDevice(device);
            device = new PSD(addr, 2);
            if (device)
               AddInstalledDevice(device);
            break;
         case DeviceTypeMVP:
            device = new MVP(addr, 1);
            if (device)
               AddInstalledDevice(device);
            break;
         case DeviceTypePSD2: case DeviceTypePSD3: case DeviceTypePSD4: case DeviceTypePSD6: case DeviceTypePSD8: 
            //FIXME all these have a valve and a syringe pump. Need a way to check whether the valve exists or not.
            device = new MVP(addr, 0);
            if (device)
               AddInstalledDevice(device);
            //Add the syringe
            device = new PSD(addr, 0);
            if (device)
               AddInstalledDevice(device);
            break;
         default:   
            err = ERR_UNKNOWN_VALVE_TYPE;
      }

   }
   return DEVICE_OK;
}


int
MVPChain::SelectChannel(char address, int index)
{
   int err = DEVICE_OK;
   if (index > 0) {
      ChannelSelectionCommand csCmd(address, index);
      err = SendRecv(csCmd);
   }   
   return err;
}
       
int
MVPChain::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(port_);
   }
   return DEVICE_OK;
}


int
MVPChain::SendRecv(HamiltonCommand& cmd)
{
   int err;

   err = SendSerialCommand(port_.c_str(), cmd.Get().c_str(), RNO_TERM);
   if (err != DEVICE_OK)
      return err;

   bool expectsMore = true;
   while (expectsMore)
   {
      std::string answer;
      err = GetSerialAnswer(port_.c_str(), RNO_TERM, answer);
      if (err != DEVICE_OK)
         return err;

      err = cmd.ParseResponse(answer, expectsMore);
      if (err != DEVICE_OK)
         return err;
   }

   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Hamilton Valve
///////////////////////////////////////////////////////////////////////////////

MVP::MVP(char address, int index) :
   lock_(), 
   address_(address),
   index_channel_(index), 
   index_E2_(1),
   valveType_(ValveTypeUnknown),
   rotationDirection_(LEAST_ANGLE)
{
}


MVP::~MVP()
{
}


void
MVP::GetName(char* name) const
{
   std::string suffix(1, address_);

   //This ensures any previous naming still applies
   //For the Microlab, this will translate to -a1 and -a2
   if (index_channel_ > 0)
       suffix += index_channel_;

   CDeviceUtils::CopyLimitedString(name,
         (DEVICE_NAME_MVP_PREFIX + suffix).c_str());
}


int
MVP::Initialize()
{
   int err;
   const char* description = NULL;
   std::string firmware;
   std::string deviceName;

   // FIXME: This here is not thread-safe, but are initializations done one at a time? Does it matter?
   err = SelectChannel();
   if (err != DEVICE_OK)
      return err;

   FirmwareVersionRequest fvReq(address_);
   err = SendRecv(fvReq);
   if (err != DEVICE_OK)
      return err;

   firmware = fvReq.GetFirmwareVersion();
   deviceType_ = GetDeviceTypeFromFirmware(firmware);
   deviceName = GetDeviceTypeName(deviceType_);
   LogMessage(("Initializing MVP device '" + std::string(1, address_) + "' " + firmware + " " + deviceName).c_str(), true);

   err = CreateStringProperty("FirmwareVersion",
         firmware.c_str(), true);
   if (err != DEVICE_OK)
      return err;

   switch (deviceType_)
   {
      case DeviceTypeML600: case DeviceTypeML500A: case DeviceTypeML500B: case DeviceTypeML500C: case DeviceTypeML700:
         // FIXME need to check these against actual hardware, I'm just assuming here they are dual pump / dual valve systems 
         // a and c = These ASCII values show errors for the left (a) and right (c) syringes.
         // b and d = These ASCII values show errors for the left (b) and right (d) valves.
         if (index_channel_ == 1) index_E2_ = 1;
         if (index_channel_ == 2) index_E2_ = 3;
         break;
      default:   
         index_E2_ = 1;
   }

   ValveErrorRequest errReq(address_,index_E2_);
   err = SendRecv(errReq);

   if (err != DEVICE_OK)
      return err;
   if (errReq.IsValveNotInitialized())
   {
      ValveInitializationCommand initCmd(address_);
      err = SendRecv(initCmd);
      if (err != DEVICE_OK)
         return err;

      MM::MMTime deadline = GetCurrentMMTime() + MM::MMTime(15 * 1000 * 1000);
      bool busy = true;
      while (busy && GetCurrentMMTime() < deadline)
      {
         busy = Busy();
         if (!busy)
            break;
         CDeviceUtils::SleepMs(200);
      }
      if (busy)
         return ERR_INITIALIZATION_TIMED_OUT;
   }

   // Description
   err = CreateStringProperty(MM::g_Keyword_Description, deviceName.c_str(), true);
   if (err != DEVICE_OK)
      return err;

   ValveTypeRequest typeReq(address_);
   err = SendRecv(typeReq);
   if (err != DEVICE_OK)
      return err;
   valveType_ = typeReq.GetValveType();
   if (valveType_ == ValveTypeUnknown)
      return ERR_UNKNOWN_VALVE_TYPE;
   err = CreateStringProperty("ValveType",
         GetValveTypeName(typeReq.GetValveType()).c_str(), true);
   if (err != DEVICE_OK)
      return err;

   long pos;
   err = GetPosition(pos);
   if (err != DEVICE_OK)
      return err;
   err = CreateIntegerProperty(MM::g_Keyword_State, pos, false,
         new CPropertyAction(this, &MVP::OnState));
   if (err != DEVICE_OK)
      return err;
   for (unsigned long i = 0; i < GetNumberOfPositions(); ++i)
   {
      char s[16];
      snprintf(s, 15, "%ld", i);
      AddAllowedValue(MM::g_Keyword_State, s);
   }

   err = CreateStringProperty(MM::g_Keyword_Label, "Undefined", false,
         new CPropertyAction(this, &CStateBase::OnLabel));
   if (err != DEVICE_OK)
      return err;
   for (unsigned long i = 0; i < GetNumberOfPositions(); ++i)
   {
      char label[32];
      snprintf(label, 31, "Position-%ld", i);
      SetPositionLabel(i, label);
   }

   ValveSpeedRequest speedReq(address_);
   err = SendRecv(speedReq);
   if (err != DEVICE_OK)
      return err;
   err = CreateIntegerProperty("ValveSpeedHz",
         speedReq.GetSpeedHz(), true);
   if (err != DEVICE_OK)
      return err;

   err = CreateStringProperty("RotationDirection",
         RotationDirectionToString(rotationDirection_).c_str(), false,
         new CPropertyAction(this, &MVP::OnRotationDirection));
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("RotationDirection", "Clockwise");
   AddAllowedValue("RotationDirection", "Counterclockwise");
   AddAllowedValue("RotationDirection", "Least rotation angle");

   return DEVICE_OK;
}


int
MVP::Shutdown()
{
   return DEVICE_OK;
}


bool
MVP::Busy()
{
   bool busy = false;

   MMThreadGuard myLock(lock_);
   {
      int err = SelectChannel();
      if (err == DEVICE_OK)
      {
         MovementFinishedRequest req(address_);
         err = SendRecv(req);
         busy = !req.IsMovementFinished();
      }
      if (err != DEVICE_OK)
         busy = false; 
   }

   return busy;
}


unsigned long
MVP::GetNumberOfPositions() const
{
   return GetValveNumberOfPositions(valveType_);
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int
MVP::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      long pos;
      int err = GetPosition(pos);
      if (err != DEVICE_OK)
         return err;
      pProp->Set(pos);
   }
   else if (eAct == MM::AfterSet)
   {
      long v;
      pProp->Get(v);
      int err = SetPosition(v);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int
MVP::OnRotationDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(RotationDirectionToString(rotationDirection_).c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string v;
      pProp->Get(v);
      rotationDirection_ = RotationDirectionFromString(v);
   }
   return DEVICE_OK;
}


int
MVP::SendRecv(HamiltonCommand& cmd)
{
   return static_cast<MVPChain*>(GetParentHub())->SendRecv(cmd);
}


int
MVP::SelectChannel()
{
   int err = DEVICE_OK; 
   if (index_channel_ > 0) 
      err = static_cast<MVPChain*>(GetParentHub())->SelectChannel(address_, index_channel_);
   return err;
}


int
MVP::GetPosition(long& pos)
{
   int err; 
   ValvePositionRequest req(address_);

   MMThreadGuard myLock(lock_);
   {
      // Channel selection
      err = SelectChannel();
      if (err == DEVICE_OK)
         err = SendRecv(req);
   }

   if (err != DEVICE_OK)
      return err;

   pos = req.GetPosition();
   return DEVICE_OK;
}


int
MVP::SetPosition(long pos)
{
   int err;
   long curPos;

   err = GetPosition(curPos);
   if (err != DEVICE_OK)
      return err;

   MMThreadGuard myLock(lock_);
   {
      // Channel selection
      err = SelectChannel();
      if (err == DEVICE_OK)
      {    
         ValvePositionCommand cmd(address_, ShouldRotateCCW(curPos, pos), pos);
         err = SendRecv(cmd);
      }
   }
   return err;
}


bool
MVP::ShouldRotateCCW(int curPos, int newPos)
{
   switch (rotationDirection_)
   {
      case CLOCKWISE:
         return false;
      case COUNTERCLOCKWISE:
         return true;
      case LEAST_ANGLE:
      default:
         {
            int cwAngle = GetValveRotationAngle(valveType_, false, curPos, newPos);
            int ccwAngle = GetValveRotationAngle(valveType_, true, curPos, newPos);
            return (ccwAngle < cwAngle);
         }
   }
}


std::string
MVP::RotationDirectionToString(RotationDirection rd)
{
   switch (rd)
   {
      case CLOCKWISE:
         return "Clockwise";
      case COUNTERCLOCKWISE:
         return "Counterclockwise";
      case LEAST_ANGLE:
      default:
         return "Least rotation angle";
   }
}


MVP::RotationDirection
MVP::RotationDirectionFromString(const std::string& s)
{
   if (s == "Clockwise")
      return CLOCKWISE;
   if (s == "Counterclockwise")
      return COUNTERCLOCKWISE;
   return LEAST_ANGLE;
}


///////////////////////////////////////////////////////////////////////////////
// Hamilton Syringe Pumps
///////////////////////////////////////////////////////////////////////////////

PSD::PSD(char address, int index) :
   lock_(), 
   address_(address),
   index_channel_(index), 
   index_E2_(0),
   volumeMax_ul_(1000),
   flowrate_ul_per_sec_(2000),
   speed_(0),
   back_off_steps_(0), 
   return_steps_(0), 
   volume_ul_(0),
   busy_(false),
   initialized_(false)
{
}


PSD::~PSD()
{
}


void
PSD::GetName(char* name) const
{
   std::string suffix(1, address_);

   //This ensures any previous naming still applies
   //For the Microlab, this will translate to -a1 and -a2
   if (index_channel_ > 0)
       suffix += index_channel_;

   CDeviceUtils::CopyLimitedString(name,
         (DEVICE_NAME_PSD_PREFIX + suffix).c_str());
}


int
PSD::Initialize()
{
   int err;
   const char* description = NULL;
   std::string firmware;
   std::string deviceName;

   // Channel selection
   err = SelectChannel();
   if (err != DEVICE_OK)
      return err;

   FirmwareVersionRequest fvReq(address_);
   err = SendRecv(fvReq);
   if (err != DEVICE_OK)
      return err;

   firmware = fvReq.GetFirmwareVersion();
   deviceType_ = GetDeviceTypeFromFirmware(firmware);
   deviceName = GetDeviceTypeName(deviceType_);
   LogMessage(("Initializing PSD device '" + std::string(1, address_) + "' " + firmware + " " + deviceName).c_str(), true);

   err = CreateStringProperty("FirmwareVersion",
         firmware.c_str(), true);
   if (err != DEVICE_OK)
      return err;

   switch (deviceType_)
   {
      case DeviceTypeML600: case DeviceTypeML500A: case DeviceTypeML500B: case DeviceTypeML500C: case DeviceTypeML700:
         // FIXME need to check these against actual hardware, I'm just assuming here they are dual pump / dual valve systems 
         // a and c = These ASCII values show errors for the left (a) and right (c) syringes.
         // b and d = These ASCII values show errors for the left (b) and right (d) valves.
         if (index_channel_ == 1) index_E2_ = 0;
         if (index_channel_ == 2) index_E2_ = 2;
         break;
      default:   
         index_E2_ = 0;
   }

   SyringeErrorRequest errReq(address_,index_E2_);
   err = SendRecv(errReq);

   if (err != DEVICE_OK)
      return err;
   if (errReq.IsSyringeNotInitialized())
   {
      err = Home(); 
      if (err != DEVICE_OK)
         return err;
   }

   /* Syringe parameters */

   SyringeResolutionRequest srReq(address_);
   err = SendRecv(srReq);
   if (err != DEVICE_OK)
      return err;
   driveResolution_ = (DriveResolution)srReq.GetResolution();

   SyringeBackOffStepsRequest bosReq(address_);
   err = SendRecv(bosReq);
   if (err != DEVICE_OK)
      return err;
   back_off_steps_ = bosReq.GetSteps();

   SyringeReturnStepsRequest rsReq(address_);
   err = SendRecv(rsReq);
   if (err != DEVICE_OK)
      return err;
   return_steps_ = rsReq.GetSteps();

   // Description
   // -----------
   err = CreateStringProperty(MM::g_Keyword_Description, deviceName.c_str(), true);
   if (err != DEVICE_OK)
      return err;

   // Resolution
   // ----------
   err = CreateStringProperty("DriveResolution",
         DriveResolutionToString(driveResolution_).c_str(), false,
         new CPropertyAction(this, &PSD::OnDriveResolution));
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("DriveResolution", g_Keyword_HalfResolution );
   AddAllowedValue("DriveResolution", g_Keyword_FullResolution );
   AddAllowedValue("DriveResolution", g_Keyword_FullResolutionDisabled );

   // Full Stroke Steps 
   // -----------------
   full_stroke_steps_ = GetSyringeThrowSteps(deviceType_, driveResolution_);
   err = CreateIntegerProperty(g_Keyword_FullStrokeSteps, full_stroke_steps_, true, 
         new CPropertyAction (this, &PSD::OnFullStrokeSteps));
   if (err != DEVICE_OK)
      return err;

   std::ostringstream os;
   os << "Resolution: " << driveResolution_ << " - Full stroke: " << full_stroke_steps_;
   LogMessage(os.str().c_str());

   // Max Volume
   // ----------
   err = CreateFloatProperty(MM::g_Keyword_Max_Volume, volumeMax_ul_, false,
         new CPropertyAction (this, &PSD::OnMaxVolume));
   if (err != DEVICE_OK)
      return err;

   // Flow rate
   // ---------
   err = CreateFloatProperty(MM::g_Keyword_Flowrate, flowrate_ul_per_sec_, false,
         new CPropertyAction (this, &PSD::OnFlowrate));
   if (err != DEVICE_OK)
      return err;

   // Volume
   // ------
   err = CreateFloatProperty(MM::g_Keyword_Current_Volume, volume_ul_, true,
         new CPropertyAction (this, &PSD::OnVolume));
   if (err != DEVICE_OK)
      return err;

   // Back-off Steps 
   // --------------
   err = CreateIntegerProperty(g_Keyword_BackOffSteps, back_off_steps_, false,
         new CPropertyAction (this, &PSD::OnBackOffSteps));
   if (err != DEVICE_OK)
      return err;

   // Return Steps 
   // ------------
   err = CreateIntegerProperty(g_Keyword_ReturnSteps, return_steps_, false,
         new CPropertyAction (this, &PSD::OnReturnSteps));
   if (err != DEVICE_OK)
      return err;

   initialized_ = true;
   return DEVICE_OK;
}


int
PSD::Shutdown()
{
   Stop();

   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool
PSD::Busy()
{
   bool busy = false;


   MMThreadGuard myLock(lock_);
   {
      // Channel selection
      int err = SelectChannel();
      if (err == DEVICE_OK)
      {
         MovementFinishedRequest req(address_);
         err = SendRecv(req);
         busy = !req.IsMovementFinished();
      }
      if (err != DEVICE_OK)
         busy = false; 
   }

   return busy;
}

int PSD::SetDriveResolution(int resolution)
{
   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringeResolutionCommand req(address_, resolution);
         err = SendRecv(req);
         if (err == DEVICE_OK)
            driveResolution_ = (DriveResolution)resolution;
      }
   }
   return err;
}

int PSD::SetReturnSteps(int steps)
{
   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringeReturnStepsCommand req(address_, steps);
         err = SendRecv(req);
         if (err == DEVICE_OK)
            return_steps_ = steps; 
      }
   }
   return err;
}

int PSD::SetBackOffSteps(int steps)
{
   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringeBackOffStepsCommand req(address_, steps);
         err = SendRecv(req);
         if (err == DEVICE_OK)
            back_off_steps_ = steps;
      }
   }
   return err;
}

/*
int PSD::Aspirate() 
{
   LogMessage("Filling up the syringe", true);
   return AspirateVolume(volumeMax_ul_ - volume_ul_);
}

int PSD::AspirateVolume(double volUl) 
{
   if (volUl < 0 || volUl > (volumeMax_ul_ - volume_ul_))
         return ERR_UNKNOWN_POSITION;

   long steps = (long)(full_stroke_steps_ * volUl / volumeMax_ul_);

   std::ostringstream os;
   os << "Aspirating : " << volUl << " uL - " << steps;
   LogMessage(os.str().c_str());

   // Channel selection
   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringePickupCommand pCmd(address_, steps, speed_);
         err = SendRecv(pCmd);
         if (err == DEVICE_OK)
            volume_ul_ += volUl;
      }
   }

   return err;
}

int PSD::Dispense() 
{
   LogMessage("Emptying the syringe", true);
   return DispenseVolume(volume_ul_);
}
*/

//Negative volume aspirates? Or error? Maybe a flag?
//Aspirate with invert direction?
int PSD::DispenseVolumeUl(double volUl) 
{
   if (volUl < 0 || volUl > volumeMax_ul_)
         return ERR_UNKNOWN_POSITION;

   long steps = (long)(full_stroke_steps_ * volUl / volumeMax_ul_);

   std::ostringstream os;
   os << "Dispensing : " << volUl << " uL - " << steps;
   LogMessage(os.str().c_str());

   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringeDispenseCommand dCmd(address_, steps, speed_);
         err = SendRecv(dCmd);
         if (err == DEVICE_OK)
            volume_ul_ -= volUl;
      }
   }
   return err;
}

int PSD::DispenseDurationSeconds(double durSec)
{
   double volUl, flowrate;
   int nRet = DEVICE_OK;

   nRet = GetFlowrateUlPerSecond(flowrate);

   if (nRet != DEVICE_OK )
      return nRet;

   volUl = durSec * flowrate;

   if (volUl < 0 || volUl > volumeMax_ul_)
         return ERR_UNKNOWN_POSITION;

   return DispenseVolumeUl(volUl);
}

int PSD::Home() 
{
   LogMessage("Homing the syringe pump", true);

   int err;
   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         SyringeInitializationCommand initCmd(address_);
         err = SendRecv(initCmd);
      }
   }
   if (err != DEVICE_OK)
      return err;

   MM::MMTime deadline = GetCurrentMMTime() + MM::MMTime(15 * 1000 * 1000);
   bool busy = true;
   while (busy && GetCurrentMMTime() < deadline)
   {
      busy = Busy();
      if (!busy)
         break;
      CDeviceUtils::SleepMs(200);
   }
   if (busy)
      return ERR_INITIALIZATION_TIMED_OUT;

   LogMessage("Did Home the syringe pump", true);
   return DEVICE_OK;
}

//FIXME I need to check this: Initiate a dispense, followed by as many start/stop as needed? Wait for start? Or Autostart?
int PSD::Start() 
{
   int err;
   LogMessage("Starting the pump", true);

   MMThreadGuard myLock(lock_);
   {
   }

   LogMessage("Did Start the pump", true);
   return DEVICE_OK;
}

int PSD::Stop() 
{
   int err;
   LogMessage("Stopping the pump", true);

   MMThreadGuard myLock(lock_);
   {
      err = SelectChannel();
      if (err == DEVICE_OK)
      {
         // This pauses the pump
         SyringeHaltCommand haltCmd(address_);
         err = SendRecv(haltCmd);
      }
      if (err == DEVICE_OK)
      {
         // This clears the pause state (so it stops)
         SyringeClearPendingCommand clearCmd(address_);
         err = SendRecv(clearCmd);
      }
   }

   if (err != DEVICE_OK)
      return err;

   LogMessage("Did Stop the pump", true);
   return DEVICE_OK;
}

int
PSD::SendRecv(HamiltonCommand& cmd)
{
   return static_cast<MVPChain*>(GetParentHub())->SendRecv(cmd);
}

int
PSD::SelectChannel()
{
   int err = DEVICE_OK; 
   if (index_channel_ > 0) 
      err = static_cast<MVPChain*>(GetParentHub())->SelectChannel(address_, index_channel_);
   return err;
}

std::string
PSD::DriveResolutionToString(DriveResolution dr)
{
   switch (dr)
   {
      case FULLRESOLUTION:
         return g_Keyword_FullResolution;
      case FULLRESOLUTION_OVERLOADDISABLED:
         return g_Keyword_FullResolutionDisabled;
      default:
         return g_Keyword_HalfResolution;
   }
}


PSD::DriveResolution
PSD::DriveResolutionFromString(const std::string& s)
{
   if (s == g_Keyword_FullResolution)
      return FULLRESOLUTION;
   if (s == g_Keyword_FullResolutionDisabled)
      return FULLRESOLUTION_OVERLOADDISABLED;
   return HALFRESOLUTION;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int PSD::OnDriveResolution(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DriveResolutionToString(driveResolution_).c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string v;
      pProp->Get(v);
      driveResolution_ = DriveResolutionFromString(v);

      full_stroke_steps_ = GetSyringeThrowSteps(deviceType_, driveResolution_);
      OnPropertyChanged(g_Keyword_FullStrokeSteps, CDeviceUtils::ConvertToString(full_stroke_steps_));
   }
   return DEVICE_OK;
}

int PSD::OnFullStrokeSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << full_stroke_steps_;
      pProp->Set(s.str().c_str());
   }
   return DEVICE_OK;
}

int PSD::OnVolume(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int nRet = DEVICE_OK;
   if (eAct == MM::BeforeGet)
   {
      double volume; 
      nRet = GetVolumeUl(volume);
      if (nRet != DEVICE_OK)
         return nRet; 

      std::stringstream s;
      s << volume;
      pProp->Set(s.str().c_str());
   }
   return DEVICE_OK;
}

int PSD::OnMaxVolume(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int nRet = DEVICE_OK;
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << volumeMax_ul_;
      pProp->Set(s.str().c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      double vol;
      pProp->Get(vol);
      nRet = SetMaxVolumeUl(vol);
   }
   return nRet;
}

int PSD::OnReturnSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int nRet = DEVICE_OK;
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << return_steps_;
      pProp->Set(s.str().c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      long steps;
      pProp->Get(steps);
      nRet = SetReturnSteps(steps);
   }
   return nRet;
}

int PSD::OnBackOffSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int nRet = DEVICE_OK;
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << back_off_steps_;
      pProp->Set(s.str().c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      long steps;
      pProp->Get(steps);
      nRet = SetBackOffSteps(steps);
   }
   return nRet;
}

int PSD::OnFlowrate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   int nRet = DEVICE_OK;
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << flowrate_ul_per_sec_;
      pProp->Set(s.str().c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      double fr;
      pProp->Get(fr);
      nRet = SetFlowrateUlPerSecond(fr);
   }
   return nRet;
}

