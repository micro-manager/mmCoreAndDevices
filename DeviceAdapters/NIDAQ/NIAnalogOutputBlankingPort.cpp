// DESCRIPTION:   Drive multiple analog and digital outputs on NI DAQ
//                Analog-output blanking port: gate a set of AO channels with a
//                hardware trigger, without per-frame sequencing.
// AUTHOR:        Alex Landolt, 2026
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation; either
//                version 2.1 of the License, or (at your option) any later
//                version.
//
//                This library is distributed in the hope that it will be
//                useful, but WITHOUT ANY WARRANTY; without even the implied
//                warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//                PURPOSE.  See the GNU Lesser General Public License for more
//                details.
//
//                You should have received a copy of the GNU Lesser General
//                Public License along with this library; if not, write to the
//                Free Software Foundation, Inc., 51 Franklin Street, Fifth
//                Floor, Boston, MA  02110-1301  USA

#include "NIDAQ.h"

#include "ModuleInterface.h"

#include <climits>
#include <sstream>


AnalogOutputBlankingPort::AnalogOutputBlankingPort(const std::string& deviceName) :
   ErrorTranslator(24000, 24999, &AnalogOutputBlankingPort::SetErrorText),
   deviceName_(deviceName),
   firstLine_(0),
   numLines_(8),
   channelNamesSpec_(""),
   inputLine_(0),
   numChannels_(0),
   initialized_(false),
   blanking_(false),
   blankOnLow_(true),
   open_(true),
   onVolts_(5.0),
   offVolts_(0.0),
   minVolts_(0.0),
   maxVolts_(5.0),
   pos_(0),
   numPos_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_VOLTAGE_OUT_OF_RANGE, "Requested voltage is out of range.");
   SetErrorText(ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS, "Requested voltage range exceeds the device's limits.");

   // Contiguous block of AO lines: first line + count.
   CPropertyAction* pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnFirstLine);
   CreateIntegerProperty("First AO Line", firstLine_, false, pAct, true);
   SetPropertyLimits("First AO Line", 0, 31);

   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnNumLines);
   CreateIntegerProperty("Number of AO Lines", numLines_, false, pAct, true);
   SetPropertyLimits("Number of AO Lines", 1, maxLines_);

   // Optional names for the per-channel properties, ';'-separated (commas are
   // the .cfg delimiter), e.g. "Laser405;Laser488;LED".  Unnamed channels: "aoN".
   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnChannelNames);
   CreateStringProperty("Channel Names", channelNamesSpec_.c_str(), false, pAct, true);

   // Change detection is only available on the hardware-timed port0.
   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnTriggerInputLine);
   CreateIntegerProperty("Trigger Input Line (port0)", inputLine_, false, pAct, true);
   SetPropertyLimits("Trigger Input Line (port0)", 0, 31);
}


AnalogOutputBlankingPort::~AnalogOutputBlankingPort()
{
   Shutdown();
}


int AnalogOutputBlankingPort::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   int err = ParseChannelSpec();
   if (err != DEVICE_OK)
      return err;

   err = GetHub()->GetVoltageLimits(minVolts_, maxVolts_);
   if (err != DEVICE_OK)
      return TranslateHubError(err);

   triggerTerminal_ = deviceName_ + "/port0/line" + std::to_string(inputLine_);

   CPropertyAction* pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnOnVoltage);
   err = CreateFloatProperty("On Voltage", onVolts_, false, pAct);
   if (err != DEVICE_OK)
      return err;
   SetPropertyLimits("On Voltage", minVolts_, maxVolts_);

   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnOffVoltage);
   err = CreateFloatProperty("Off Voltage", offVolts_, false, pAct);
   if (err != DEVICE_OK)
      return err;
   SetPropertyLimits("Off Voltage", minVolts_, maxVolts_);

   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnBlanking);
   err = CreateStringProperty("Blanking", blanking_ ? g_On : g_Off, false, pAct);
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("Blanking", g_Off);
   AddAllowedValue("Blanking", g_On);

   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnBlankingTriggerDirection);
   err = CreateStringProperty("Blank on", blankOnLow_ ? g_Low : g_High, false, pAct);
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("Blank on", g_Low);
   AddAllowedValue("Blank on", g_High);

   err = CreateStringProperty("TriggerInputPin", triggerTerminal_.c_str(), true);
   if (err != DEVICE_OK)
      return err;

   // Diagnostic: trigger edges clocked by the AO task since blanking started.
   // Expect 2 per frame; odd while the camera is idle means an inverted phase.
   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnTriggerEdgeCount);
   err = CreateIntegerProperty("Trigger Edge Count", 0, true, pAct);
   if (err != DEVICE_OK)
      return err;

   // State: bit i set == channel i active.  numChannels_ <= maxLines_ (31).
   numPos_ = (numChannels_ >= 31) ? LONG_MAX : (1L << numChannels_) - 1;
   pAct = new CPropertyAction(this, &AnalogOutputBlankingPort::OnState);
   err = CreateIntegerProperty("State", 0, false, pAct);
   if (err != DEVICE_OK)
      return err;
   SetPropertyLimits("State", 0, numPos_);

   std::vector<std::string> names;
   {
      std::stringstream ns(channelNamesSpec_);
      std::string tok;
      while (std::getline(ns, tok, ';'))
      {
         size_t b = tok.find_first_not_of(" \t");
         size_t e = tok.find_last_not_of(" \t");
         names.push_back(b == std::string::npos ? std::string() : tok.substr(b, e - b + 1));
      }
   }

   // Per-channel 0/1 properties, each toggling one State bit.
   for (size_t i = 0; i < channelLines_.size(); ++i)
   {
      std::string propName;
      if (i < names.size() && !names[i].empty())
         propName = names[i];
      else
      {
         std::ostringstream os;
         os << "ao" << channelLines_[i];
         propName = os.str();
      }
      CPropertyActionEx* pActEx = new CPropertyActionEx(this, &AnalogOutputBlankingPort::OnChannelLine, static_cast<long>(i));
      err = CreateIntegerProperty(propName.c_str(), 0, false, pActEx);
      if (err != DEVICE_OK)
         return err;
      SetPropertyLimits(propName.c_str(), 0, 1);
   }

   // Gate Closed Position
   err = CreateProperty(MM::g_Keyword_Closed_Position, "0", MM::Integer, false);
   if (err != DEVICE_OK)
      return err;
   GetGateOpen(open_);

   // In case someone left some lines on:
   err = ApplyState(0);
   if (err != DEVICE_OK)
      return TranslateHubError(err);

   initialized_ = true;
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   GetHub()->StopAOBlanking();
   blanking_ = false;

   // Leave the outputs at the off voltage (AO holds its value after the clear).
   GetHub()->SetAOPortState(physChannelList_, numChannels_, onVolts_, offVolts_, 0);
   GetHub()->StopAOBlanking();

   initialized_ = false;
   return DEVICE_OK;
}


void AnalogOutputBlankingPort::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name,
      (g_DeviceNameNIDAQAOBlankPrefix + deviceName_).c_str());
}


// Expand firstLine_/numLines_ into the line list and the DAQmx channel string.
int AnalogOutputBlankingPort::ParseChannelSpec()
{
   physChannelList_.clear();
   channelLines_.clear();
   numChannels_ = 0;

   if (numLines_ < 1 || numLines_ > maxLines_ || firstLine_ < 0)
      return DEVICE_INVALID_PROPERTY_VALUE;

   std::ostringstream physList;
   for (long i = 0; i < numLines_; ++i)
   {
      long line = firstLine_ + i;
      if (i > 0)
         physList << ",";
      physList << deviceName_ << "/ao" << line;
      channelLines_.push_back(line);
      ++numChannels_;
   }
   physChannelList_ = physList.str();
   return DEVICE_OK;
}


long AnalogOutputBlankingPort::GatedState(long state)
{
   bool gateOpen;
   GetGateOpen(gateOpen);
   if (gateOpen)
      return state;
   long closedState;
   GetProperty(MM::g_Keyword_Closed_Position, closedState);
   return closedState;
}


int AnalogOutputBlankingPort::ApplyState(long state)
{
   // With blanking on the trigger gates every exposure, so the shutter gate is
   // not applied; the light is off whenever the camera is idle.
   if (blanking_)
      return GetHub()->StartAOBlanking(physChannelList_, numChannels_, onVolts_, offVolts_,
         state, blankOnLow_, triggerTerminal_);

   return GetHub()->SetAOPortState(physChannelList_, numChannels_, onVolts_, offVolts_,
      GatedState(state));
}


int AnalogOutputBlankingPort::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(pos_);
   }
   else if (eAct == MM::AfterSet)
   {
      bool gateOpen;
      GetGateOpen(gateOpen);
      long pos;
      pProp->Get(pos);
      if ((pos == pos_) && (open_ == gateOpen))
         return DEVICE_OK;

      // With blanking on, a gate change alone (AutoShutter, via SetGateOpen ->
      // SetPosition) must not restart the tasks: a restart while the camera is
      // pulsing can miss an edge and latch the phase inverted.
      if (!blanking_ || pos != pos_)
      {
         int err = ApplyState(pos);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
      }

      pos_ = pos;
      open_ = gateOpen;
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(blanking_ ? g_On : g_Off);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string response;
      pProp->Get(response);
      bool blanking = (response == g_On);
      if (blanking_ == blanking)
         return DEVICE_OK;

      if (blanking)
      {
         int err = GetHub()->StartAOBlanking(physChannelList_, numChannels_, onVolts_, offVolts_,
            pos_, blankOnLow_, triggerTerminal_);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
         blanking_ = true;
      }
      else
      {
         int err = GetHub()->StopAOBlanking();
         if (err != DEVICE_OK)
            return TranslateHubError(err);
         blanking_ = false;
         err = ApplyState(pos_);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
      }
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(blankOnLow_ ? g_Low : g_High);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string response;
      pProp->Get(response);
      bool blankOnLow = (response == g_Low);
      if (blankOnLow_ == blankOnLow)
         return DEVICE_OK;

      blankOnLow_ = blankOnLow;
      if (blanking_)
      {
         int err = ApplyState(pos_);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
      }
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnOnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(onVolts_);
   }
   else if (eAct == MM::AfterSet)
   {
      double v;
      pProp->Get(v);
      if (v < minVolts_ || v > maxVolts_)
         return ERR_VOLTAGE_OUT_OF_RANGE;
      onVolts_ = v;
      if (initialized_)
      {
         int err = ApplyState(pos_);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
      }
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnOffVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(offVolts_);
   }
   else if (eAct == MM::AfterSet)
   {
      double v;
      pProp->Get(v);
      if (v < minVolts_ || v > maxVolts_)
         return ERR_VOLTAGE_OUT_OF_RANGE;
      offVolts_ = v;
      if (initialized_)
      {
         int err = ApplyState(pos_);
         if (err != DEVICE_OK)
            return TranslateHubError(err);
      }
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnTriggerEdgeCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      uInt64 count;
      int err = GetHub()->GetAOBlankingEdgeCount(count);
      if (err != DEVICE_OK)
         return TranslateHubError(err);
      pProp->Set(static_cast<long>(count));
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnChannelLine(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((pos_ >> index) & 1L);
   }
   else if (eAct == MM::AfterSet)
   {
      long val;
      pProp->Get(val);
      long pos = pos_;
      if (val)
         pos |= (1L << index);
      else
         pos &= ~(1L << index);
      std::ostringstream os;
      os << pos;
      return SetProperty("State", os.str().c_str());
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnFirstLine(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
      pProp->Set(firstLine_);
   else if (eAct == MM::AfterSet)
      pProp->Get(firstLine_);
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnNumLines(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
      pProp->Set(numLines_);
   else if (eAct == MM::AfterSet)
      pProp->Get(numLines_);
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnChannelNames(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
      pProp->Set(channelNamesSpec_.c_str());
   else if (eAct == MM::AfterSet)
      pProp->Get(channelNamesSpec_);
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::OnTriggerInputLine(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(inputLine_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(inputLine_);
   }
   return DEVICE_OK;
}


int AnalogOutputBlankingPort::TranslateHubError(int err)
{
   if (err == DEVICE_OK)
      return DEVICE_OK;
   char buf[MM::MaxStrLength];
   if (GetHub()->GetErrorText(err, buf))
      return NewErrorCode(buf);
   return NewErrorCode("Unknown hub error");
}
