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

#include "HamiltonMVP.h"
#include "HamiltonPSD.h"

#include "MMDeviceConstants.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf _snprintf
#endif


const char* const RNO_TERM = "\r";
const char RNO_ACK = 6;
const char RNO_NAK = 21;

enum {
   ERR_ECHO_MISMATCH = 20001,
   ERR_NAK,
   ERR_UNEXPECTED_RESPONSE,
   ERR_RESPONSE_PARITY,
};

class HamiltonCommand
{
protected:
   bool TestBit(char byte, int bit)
   {
      return (byte & (1 << bit)) != 0;
   }

   int ParseDecimal(const std::string& s, int maxNDigits, int& result)
   {
      if (s.empty() || s.size() > (unsigned int) maxNDigits)
         return ERR_UNEXPECTED_RESPONSE;
      for (unsigned int i = 0; i < s.size(); ++i)
      {
         char c = s[i];
         if (c < '0' || c > '9')
            return ERR_UNEXPECTED_RESPONSE;
      }
      result = std::atoi(s.c_str());
      return DEVICE_OK;
   }

   std::string FormatDecimal(int /* nDigits */, int value)
   {
      // It turns out that there is no need to zero-pad to a fixed number of
      // digits, despite what the manual may seem to imply.
      char buf[16];
      snprintf(buf, 15, "%d", value);
      return std::string(buf);
   }

public:

   virtual ~HamiltonCommand() {}

   virtual std::string Get() = 0;

   // expectsMore set to true if further (CR-deliminated) response should be parsed
   virtual int ParseResponse(const std::string& response, bool& expectsMore) = 0;
};


class AutoAddressingCommand : public HamiltonCommand
{
   char echoedAddr_;
   int responsesParsed_;

public:
   AutoAddressingCommand() :
      echoedAddr_('\0'),
      responsesParsed_(0)
   {}

   virtual std::string Get() { return "1a"; }
   virtual int ParseResponse(const std::string& response, bool& expectsMore)
   {
      expectsMore = false;
      if (responsesParsed_++ == 0)
      {
         if (response.size() != 2)
            return ERR_UNEXPECTED_RESPONSE;
         if (response[0] != '1')
            return ERR_UNEXPECTED_RESPONSE;
         echoedAddr_ = response[1];

         if (echoedAddr_ == 'a')
         {
            // We get the echo '1a' when the address had already been assigned.
            // In this case only, there will be a further ACK response.
            expectsMore = true;
         }

         return DEVICE_OK;
      }

      if (response.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;
      char ack = response[0];
      if (ack == RNO_NAK)
         return ERR_NAK;
      if (ack != RNO_ACK)
         return ERR_UNEXPECTED_RESPONSE;

      return DEVICE_OK;
   }

   bool HasMaxAddr() { return echoedAddr_ > 'a'; }
   char GetMaxAddr() { return echoedAddr_ - 1; }
};


class NormalCommand : public HamiltonCommand
{
   char address_;
   int responsesParsed_;

protected:
   virtual std::string GetCommandString() = 0;
   virtual int ParseContent(const std::string& content) = 0;

public:
   NormalCommand(char address) :
      address_(address),
      responsesParsed_(0)
   {}

   virtual std::string Get()
   { return std::string(1, address_) + GetCommandString(); }

   virtual int ParseResponse(const std::string& response, bool& expectsMore)
   {
      expectsMore = false;
      if (responsesParsed_++ == 0)
      {
         // The first response should be an exact echo
         if (response != Get())
            return ERR_ECHO_MISMATCH;
         expectsMore = true;
         return DEVICE_OK;
      }

      // The second response ACK or NAK; ACK may also be followed by query
      // result.
      if (response.empty())
         return ERR_UNEXPECTED_RESPONSE;
      char ack = response[0];
      if (ack == RNO_NAK)
         return ERR_NAK;
      if (ack != RNO_ACK)
         return ERR_UNEXPECTED_RESPONSE;
      std::string content = response.substr(1);
      return ParseContent(content);
   }
};


// Commands whose ACK contains no data
class NonQueryCommand : public NormalCommand
{
protected:
   virtual int ParseContent(const std::string& content)
   { return content.empty() ? DEVICE_OK : ERR_UNEXPECTED_RESPONSE; }

public:
   NonQueryCommand(char address) :
      NormalCommand(address)
   {}
};


class ValveInitializationCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "LXR"; }

public:
   ValveInitializationCommand(char address) :
      NonQueryCommand(address)
   {}
};


class SyringeInitializationCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "X1R"; }

public:
   SyringeInitializationCommand(char address) :
      NonQueryCommand(address)
   {}
};


class SyringeHaltCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "K"; }

public:
   SyringeHaltCommand(char address) :
      NonQueryCommand(address)
   {}
};


class SyringeResumeCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "$"; }

public:
   SyringeResumeCommand(char address) :
      NonQueryCommand(address)
   {}
};


class SyringeClearPendingCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "V"; }

public:
   SyringeClearPendingCommand(char address) :
      NonQueryCommand(address)
   {}
};


class ValvePositionCommand : public NonQueryCommand
{
   bool ccw_;
   int positionOneBased_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("LP") + (ccw_ ? "1" : "0") +
         FormatDecimal(2, positionOneBased_) + "R";
   }

public:
   ValvePositionCommand(char address, bool counterclockwise, long position) :
      NonQueryCommand(address),
      ccw_(counterclockwise),
      positionOneBased_(position + 1)
   {}
};


class SyringePickupCommand : public NonQueryCommand
{
   int numsteps_;
   int speed_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("P") + FormatDecimal(4, numsteps_) +
          "S" + FormatDecimal(2, speed_) + "R";
   }

public:
   SyringePickupCommand(char address, long numsteps, int speed) :
      NonQueryCommand(address),
      numsteps_(numsteps),
      speed_(speed)
   {}
};


class SyringeResolutionCommand : public NonQueryCommand
{
   int resolution_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("YSM") + FormatDecimal(1, resolution_);
   }

public:
   SyringeResolutionCommand(char address, int resolution) :
      NonQueryCommand(address),
      resolution_(resolution)
   {}
};


class SyringeSpeedCommand : public NonQueryCommand
{
   int speed_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("YSS") + FormatDecimal(2, speed_);
   }

public:
   SyringeSpeedCommand(char address, int speed) :
      NonQueryCommand(address),
      speed_(speed)
   {}
};


class SyringeReturnStepsCommand : public NonQueryCommand
{
   int return_steps_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("YSN") + FormatDecimal(4, return_steps_);
   }

public:
   SyringeReturnStepsCommand(char address, int return_steps) :
      NonQueryCommand(address),
      return_steps_(return_steps)
   {}
};


class SyringeBackOffStepsCommand : public NonQueryCommand
{
   int back_off_steps_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("YSB") + FormatDecimal(4,back_off_steps_);
   }

public:
   SyringeBackOffStepsCommand(char address, int back_off_steps) :
      NonQueryCommand(address),
      back_off_steps_(back_off_steps)
   {}
};


class SyringeDispenseCommand : public NonQueryCommand
{
   int numsteps_;
   int speed_;

protected:
   virtual std::string GetCommandString()
   {
      return std::string("D") + FormatDecimal(4, numsteps_) +
          "S" + FormatDecimal(2, speed_) + "R";
   }

public:
   SyringeDispenseCommand(char address, long numsteps, int speed) :
      NonQueryCommand(address),
      numsteps_(numsteps),
      speed_(speed)
   {}
};


class ChannelSelectionCommand : public NonQueryCommand
{
   int channel_;

protected:
   virtual std::string GetCommandString()
   {
      if (channel_ < 1 || channel_ > 2)
         channel_ = 1;

      //char chan = 'A'+channel_; 
      //return std::string("")+chan;
      return std::string(1, static_cast<char>('A' + channel_));
   }

public:
   ChannelSelectionCommand(char address, int channel) :
      NonQueryCommand(address),
      channel_(channel)
   {}
};


class ResetInstrumentCommand : public NonQueryCommand
{
protected:
   virtual std::string GetCommandString() { return "!"; }
public:
   ResetInstrumentCommand(char address) :
      NonQueryCommand(address)
   {}
};


class InstrumentStatusRequest : public NormalCommand
{
   char b1_;

protected:
   virtual std::string GetCommandString() { return "E1"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;

      b1_ = content[0];
      if (TestBit(b1_, 5))
         return ERR_UNEXPECTED_RESPONSE;
      if (!TestBit(b1_, 6))
         return ERR_UNEXPECTED_RESPONSE;
      return DEVICE_OK;
   }

public:
   InstrumentStatusRequest(char address) :
      NormalCommand(address),
      b1_(0)
   {}

   bool IsReceivedCommandButNotExecuted() { return TestBit(b1_, 0); }
   bool IsSyringeDriveBusy() { return TestBit(b1_, 1); }
   bool IsValveDriveBusy() { return TestBit(b1_, 2); }
   bool IsSyntaxError() { return TestBit(b1_, 3); }
   bool IsInstrumentError() { return TestBit(b1_, 4); }
};


class ValveErrorRequest : public NormalCommand
{
   char b1_;
   int index_;

protected:
   virtual std::string GetCommandString() { return "E2"; }
   virtual int ParseContent(const std::string& content)
   {
      // The answer is always four bytes long
      // On an MVP device, B0, B2 and B3 are 0x50
      // On a PSD/2 device, only B2 and B3 are 0x50
      // On a microlab 600, B and D are the valve values
      if (content.size() != 4)
         return ERR_UNEXPECTED_RESPONSE;

      b1_ = content[index_];

      /*
      if (content[0] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;
      if (content[2] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;
      if (content[3] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;

      if (TestBit(b1_, 3))
         return ERR_UNEXPECTED_RESPONSE;
      if (TestBit(b1_, 4))
         return ERR_UNEXPECTED_RESPONSE;
      if (TestBit(b1_, 5))
         return ERR_UNEXPECTED_RESPONSE;
      if (!TestBit(b1_, 6))
         return ERR_UNEXPECTED_RESPONSE;
      */
      return DEVICE_OK;
   }

public:
   ValveErrorRequest(char address, int byte_index) :
      NormalCommand(address),
      b1_(0),
      index_(byte_index)
   {}

   bool IsValveNotInitialized() { return TestBit(b1_, 0); }
   bool IsValveInitializationError() { return TestBit(b1_, 1); }
   bool IsValveOverloadError() { return TestBit(b1_, 2); }
   bool IsValveExists() { return !TestBit(b1_, 4); } // This on the MicroLab 600
};

class SyringeErrorRequest : public NormalCommand
{
   char b1_;
   int index_;

protected:
   virtual std::string GetCommandString() { return "E2"; }
   virtual int ParseContent(const std::string& content)
   {
      // The answer is always four bytes long
      // On an MVP device, B0, B2 and B3 are 0x50
      // On a PSD/2 device, only B2 and B3 are 0x50
      // On a microlab 600, B and D are the valve values
      if (content.size() != 4)
         return ERR_UNEXPECTED_RESPONSE;

      b1_ = content[index_];

      /*
      if (content[0] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;
      if (content[2] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;
      if (content[3] != 0x50)
         return ERR_UNEXPECTED_RESPONSE;

      if (TestBit(b1_, 3))
         return ERR_UNEXPECTED_RESPONSE;
      if (TestBit(b1_, 4))
         return ERR_UNEXPECTED_RESPONSE;
      if (TestBit(b1_, 5))
         return ERR_UNEXPECTED_RESPONSE;
      if (!TestBit(b1_, 6))
         return ERR_UNEXPECTED_RESPONSE;
      */
      return DEVICE_OK;
   }

public:
   SyringeErrorRequest(char address, int byte_index) :
      NormalCommand(address),
      b1_(0),
      index_(byte_index)
   {}

   bool IsSyringeNotInitialized() { return TestBit(b1_, 0); }
   bool IsStrokeTooLarge() { return TestBit(b1_, 1); } // This on the MicroLab 600
   bool IsSyringeInitializationError() { return TestBit(b1_, 2); }
   bool IsSyringeOverloadError() { return TestBit(b1_, 3); }
   bool IsSyringeExists() { return !TestBit(b1_, 4); } // This on the MicroLab 600
};


class MiscellaneousDeviceStatusRequest : public NormalCommand
{
   char b1_;

protected:
   virtual std::string GetCommandString() { return "E3"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;

      b1_  = content[0];
      if (TestBit(b1_, 5))
         return ERR_UNEXPECTED_RESPONSE;
      if (!TestBit(b1_, 6))
         return ERR_UNEXPECTED_RESPONSE;
      return DEVICE_OK;
   }

public:
   MiscellaneousDeviceStatusRequest(char address) :
      NormalCommand(address),
      b1_(0)
   {}

   bool IsTimerBusy() { return TestBit(b1_, 0); }
   bool IsDiagnosticModeBusy() { return TestBit(b1_, 1); }
   bool IsEEPROMBusy() { return TestBit(b1_, 2); }
   bool IsI2CBusError() { return TestBit(b1_, 3); }
   bool IsOverTemperatureError() { return TestBit(b1_, 4); }
};


class MovementFinishedRequest : public NormalCommand
{
   char response_;

protected:
   virtual std::string GetCommandString() { return "F"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;
      response_ = content[0];
      switch (response_)
      {
         case 'N':
         case 'Y':
         case '*':
            return DEVICE_OK;
         default:
            return ERR_UNEXPECTED_RESPONSE;
      }
   }

public:
   MovementFinishedRequest(char address) :
      NormalCommand(address),
      response_('\0')
   {}

   bool IsReceivedCommandButNotExecuted() { return response_ == 'N'; }
   bool IsValveDriveBusy() { return response_ == '*'; }
   bool IsMovementFinished() { return response_ == 'Y'; }
};


class ValveOverloadedRequest : public NormalCommand
{
   char response_;

protected:
   virtual std::string GetCommandString() { return "G"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;
      response_ = content[0];
      switch (response_)
      {
         case 'N':
         case 'Y':
         case '*':
            return DEVICE_OK;
         default:
            return ERR_UNEXPECTED_RESPONSE;
      }
   }

public:
   ValveOverloadedRequest(char address) :
      NormalCommand(address),
      response_('\0')
   {}

   bool IsValveOverload() { return response_ == 'Y'; }
   bool IsValveDriveBusy() { return response_ == '*'; }
   bool IsNoError() { return response_ == 'N'; }
};


class InstrumentConfigurationRequest : public NormalCommand
{
   char response_;

protected:
   virtual std::string GetCommandString() { return "H"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;
      response_ = content[0];
      switch (response_)
      {
         case 'N':
         case 'Y':
         case '*':
            return DEVICE_OK;
         default:
            return ERR_UNEXPECTED_RESPONSE;
      }
   }

public:
   InstrumentConfigurationRequest(char address) :
      NormalCommand(address),
      response_('\0')
   {}

   bool IsOneSyringeOneValve() { return response_ == 'Y'; }
   bool IsBusy() { return response_ == '*'; }
};


class SyringeOverloadedRequest : public NormalCommand
{
   char response_;

protected:
   virtual std::string GetCommandString() { return "Z"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.size() != 1)
         return ERR_UNEXPECTED_RESPONSE;
      response_ = content[0];
      switch (response_)
      {
         case 'N':
         case 'Y':
         case '*':
            return DEVICE_OK;
         default:
            return ERR_UNEXPECTED_RESPONSE;
      }
   }

public:
   SyringeOverloadedRequest(char address) :
      NormalCommand(address),
      response_('\0')
   {}

   bool IsSyringeOverload() { return response_ == 'Y'; }
   bool IsSyringeDriveBusy() { return response_ == '*'; }
   bool IsNoError() { return response_ == 'N'; }
};


class ValvePositionRequest : public NormalCommand
{
   int positionOneBased_;

protected:
   virtual std::string GetCommandString() { return "LQP"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 2, positionOneBased_); }

public:
   ValvePositionRequest(char address) :
      NormalCommand(address),
      positionOneBased_(0)
   {}

   long GetPosition() { return long(positionOneBased_ - 1); }
};


class ValveAngleRequest : public NormalCommand
{
   int angle_;

protected:
   virtual std::string GetCommandString() { return "LQA"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 3, angle_); }

public:
   ValveAngleRequest(char address) :
      NormalCommand(address),
      angle_(-1)
   {}

   int GetAngle() { return angle_; } // 0-359
};


class ValveTypeRequest : public NormalCommand
{
private:
   int type_;

protected:
   virtual std::string GetCommandString() { return "LQT"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 1, type_); }

public:
   ValveTypeRequest(char address) :
      NormalCommand(address),
      type_(ValveTypeUnknown)
   {}

   MVPValveType GetValveType() { return MVPValveType(type_); }
};


class ValveSpeedRequest : public NormalCommand
{
   int speed_;

protected:
   virtual std::string GetCommandString() { return "LQF"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 1, speed_); }

public:
   ValveSpeedRequest(char address) :
      NormalCommand(address),
      speed_(-1)
   {}

   int GetSpeedHz()
   {
      switch (speed_)
      {
         case 0: return 30;
         case 1: return 40;
         case 2: return 50;
         case 3: return 60;
         case 4: return 70;
         case 5: return 80;
         case 6: return 90;
         case 7: return 100;
         case 8: return 110;
         case 9: return 120;
         default: return 0;
      }
   }
};


class SyringeSpeedRequest : public NormalCommand
{
   int speed_;

protected:
   virtual std::string GetCommandString() { return "YQS"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 4, speed_); }

public:
   SyringeSpeedRequest(char address) :
      NormalCommand(address),
      speed_(-1)
   {}

   // The number of seconds to move the whole stroke
   int GetSpeedSeconds()
   {
      return speed_;
   }
};


class SyringeReturnStepsRequest : public NormalCommand
{
   int return_steps_;

protected:
   virtual std::string GetCommandString() { return "YQN"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 4, return_steps_); }

public:
   SyringeReturnStepsRequest(char address) :
      NormalCommand(address),
      return_steps_(-1)
   {}

   //THe number of seconds to move the whole stroke
   int GetSteps()
   {
      return return_steps_;
   }
};


class SyringeBackOffStepsRequest : public NormalCommand
{
   int back_off_steps_;

protected:
   virtual std::string GetCommandString() { return "YQB"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 4, back_off_steps_); }

public:
   SyringeBackOffStepsRequest(char address) :
      NormalCommand(address),
      back_off_steps_(-1)
   {}

   // Syringe Back-off Steps Request/Response
   int GetSteps()
   {
      return back_off_steps_;
   }
};


class SyringeResolutionRequest : public NormalCommand
{
   int resolution_;

protected:
   virtual std::string GetCommandString() { return "YQM"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 1, resolution_); }

public:
   SyringeResolutionRequest(char address) :
      NormalCommand(address),
      resolution_(-1)
   {}

   // Syringe Resolution Request/Response
   // x = 0 Half resolution
   // x = 1 Full resolution
   // x = 2 Full resolution, overload detection disabled
   int GetResolution()
   {
      return resolution_;
   }
};


class SyringePositionRequest : public NormalCommand
{
   int position_;

protected:
   virtual std::string GetCommandString() { return "YQP"; }
   virtual int ParseContent(const std::string& content)
   { return ParseDecimal(content, 4, position_); }

public:
   SyringePositionRequest(char address) :
      NormalCommand(address),
      position_(-1)
   {}

   // The position of the stroke in steps
   int GetPosition()
   {
      return position_;
   }
};


class FirmwareVersionRequest : public NormalCommand
{
   std::string version_;

protected:
   virtual std::string GetCommandString() { return "U"; }
   virtual int ParseContent(const std::string& content)
   {
      if (content.empty())
         return ERR_UNEXPECTED_RESPONSE;
      version_ = content;
      return DEVICE_OK;
   }

public:
   FirmwareVersionRequest(char address) :
      NormalCommand(address)
   {}

   std::string GetFirmwareVersion() { return version_; }
};
