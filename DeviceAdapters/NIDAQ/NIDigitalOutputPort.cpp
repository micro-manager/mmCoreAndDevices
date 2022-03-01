// DESCRIPTION:   Drive multiple analog outputs on NI DAQ
// AUTHOR:        Mark Tsuchida, 2015, Nico Stuurman 2022
// COPYRIGHT:     2015-2016, Open Imaging, Inc., 2022 Altos Labs
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



DigitalOutputPort::DigitalOutputPort(const std::string& port) :
   ErrorTranslator(21000, 21999, &DigitalOutputPort::SetErrorText),
   niPort_(port),
   initialized_(false),
   sequenceRunning_(false),
   blanking_(false),
   blankOnLow_(true),
   pos_(0),
   numPos_(0),
   portWidth_(0),
   nrOfStateSliders_(8),
   neverSequenceable_(false),
   supportsBlankingAndSequencing_(false),
   task_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_SEQUENCE_RUNNING, "A sequence is running on this port.  Please stop this sequence first.");
   SetErrorText(ERR_SEQUENCE_TOO_LONG, "Sequence is too long. Try increasing sequence length in the Hub device.");
   SetErrorText(ERR_SEQUENCE_ZERO_LENGTH, "Sequence has length zero.");
   SetErrorText(ERR_UNKNOWN_PINS_PER_PORT, "Only 8, 16 and 32 pin ports are supported.");

   CPropertyAction* pAct = new CPropertyAction(this, &DigitalOutputPort::OnSequenceable);
   CreateStringProperty("Sequencing", g_UseHubSetting, false, pAct, true);
   AddAllowedValue("Sequencing", g_UseHubSetting);
   AddAllowedValue("Sequencing", g_Never);
 
   int maxNrOfStateSliders = 8;
   int32 nierr = DAQmxGetPhysicalChanDOPortWidth(niPort_.c_str(), &portWidth_);
   // ignore errors at this point
   if (nierr == 0)
   {
      maxNrOfStateSliders = portWidth_;
   }
   pAct = new CPropertyAction(this, &DigitalOutputPort::OnNrOfStateSliders);
   CreateIntegerProperty("Nr of TTL Sliders", nrOfStateSliders_, false, pAct, true);
   SetPropertyLimits("Nr of TTL Sliders", 0, maxNrOfStateSliders);
}


DigitalOutputPort::~DigitalOutputPort()
{
   Shutdown();
}


int DigitalOutputPort::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Need to set all pins of the port to output pins here on in Hub
   int32 nierr = DAQmxGetPhysicalChanDOPortWidth(niPort_.c_str(), &portWidth_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }

   std::string tmpTriggerTerminal = niPort_ + "/line" + std::to_string(portWidth_ - 1);
   std::string tmpNiPort = niPort_ + "/line" + "0:" + std::to_string(portWidth_ - 2);
   if (GetHub()->StartDOBlankingAndOrSequence(tmpNiPort, portWidth_, true, false, 0, false, tmpTriggerTerminal) == DEVICE_OK)
      supportsBlankingAndSequencing_ = true;
   GetHub()->StopDOBlankingAndSequence(portWidth_);

   CPropertyAction* pAct;
   if (supportsBlankingAndSequencing_)
   {
      numPos_ = (1 << (portWidth_ - 1)) - 1;
      triggerTerminal_ = niPort_ + "/line" + std::to_string(portWidth_ - 1);
      niPort_ = niPort_ + "/line" + "0:" + std::to_string(portWidth_ - 2);
      pAct = new CPropertyAction(this, &DigitalOutputPort::OnBlanking);
      CreateStringProperty("Blanking", blanking_ ? g_On : g_Off, false, pAct);
      AddAllowedValue("Blanking", g_Off);
      AddAllowedValue("Blanking", g_On);
            
      CreateProperty("TriggerInputPin", triggerTerminal_.c_str(), MM::String, true);

      pAct = new CPropertyAction(this, &DigitalOutputPort::OnBlankingTriggerDirection);
      CreateStringProperty("Blank on", blankOnLow_ ? g_Low : g_High, false, pAct);
      AddAllowedValue("Blank on", g_Low);
      AddAllowedValue("Blank on", g_High);
   }
   else
   {
      numPos_ = (1 << portWidth_) - 1;
   }

   pAct = new CPropertyAction(this, &DigitalOutputPort::OnState);
   CreateIntegerProperty("State", 0, false, pAct);
   SetPropertyLimits("State", 0, numPos_);

   // In case someone left some pins high:
   SetState(0);

   for (long ttlNr = 1; ttlNr <= nrOfStateSliders_; ttlNr++)
   {
     
      std::ostringstream os;
      os << "TTL-" << ttlNr;
      std::string propName = os.str();
      CPropertyActionEx* pActEx = new CPropertyActionEx(this, &DigitalOutputPort::OnTTL, ttlNr - 1);
      CreateProperty(propName.c_str(), "0", MM::Integer, false, pActEx, false);
      SetPropertyLimits(propName.c_str(), 0, 1);
   }

   initialized_ = true;

   return DEVICE_OK;
}


int DigitalOutputPort::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   SetState(0);
   int err = StopTask();

   initialized_ = false;
   return err;
}


void DigitalOutputPort::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name,
      (g_DeviceNameNIDAQDOPortPrefix + niPort_).c_str());
}


int DigitalOutputPort::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(pos_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      long pos;
      pProp->Get(pos);
      // pause blanking, otherwise most cards will error
      int err;
      if (blanking_)
      {
         err = GetHub()->StopDOBlankingAndSequence(portWidth_);
         if (err == DEVICE_OK)
            err = GetHub()->StartDOBlankingAndOrSequence(niPort_, portWidth_, true, false, pos, blankOnLow_, triggerTerminal_);
      }
      else
      {
         err = SetState(pos);
      }
      if (err == DEVICE_OK)
         pos_ = pos;

      return err;
   }

   else if (eAct == MM::IsSequenceable)
   {
      bool sequenceable = false;
      if (supportsBlankingAndSequencing_)
      {
         bool isHubSequenceable;
         GetHub()->IsSequencingEnabled(isHubSequenceable);
         sequenceable = neverSequenceable_ ? false : isHubSequenceable;
      }

      if (sequenceable)
      {
         long maxLength;
         GetHub()->GetSequenceMaxLength(maxLength);
         pProp->SetSequenceable(maxLength);
      }
      else
      {
         pProp->SetSequenceable(0);
      }
   }
   else if (eAct == MM::AfterLoadSequence)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      std::vector<std::string> sequence = pProp->GetSequence();
      long maxLength;
      GetHub()->GetSequenceMaxLength(maxLength);
      if (sequence.size() > maxLength)
         return DEVICE_SEQUENCE_TOO_LARGE;

      if (portWidth_ == 8)
      {
         sequence8_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt8 val;
            os >> val;
            sequence8_.push_back(val);
         }
         GetHub()->getDOHub8()->RemoveDOPortFromSequencing(niPort_);
         return GetHub()->getDOHub8()->AddDOPortToSequencing(niPort_, sequence8_);
      }
      else if (portWidth_ == 16)
      {
         sequence16_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt16 val;
            os >> val;
            sequence16_.push_back(val);
         }
         GetHub()->getDOHub16()->RemoveDOPortFromSequencing(niPort_);
         return GetHub()->getDOHub16()->AddDOPortToSequencing(niPort_, sequence16_);
      }
      else if (portWidth_ == 32)
      {
         sequence32_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt32 val;
            os >> val;
            sequence32_.push_back(val);
         }
         GetHub()->getDOHub32()->RemoveDOPortFromSequencing(niPort_);
         return GetHub()->getDOHub32()->AddDOPortToSequencing(niPort_, sequence32_);
      }
   }
   else if (eAct == MM::StartSequence)
   {
      int err = DEVICE_OK;
      sequenceRunning_ = true;
      // TODO: set the first state of the sequence before we start?

      if (err == DEVICE_OK)
      {
         err = GetHub()->StopDOBlankingAndSequence(portWidth_);
         if (err != DEVICE_OK)
         {
            sequenceRunning_ = false;
            return err;
         }
         err = GetHub()->StartDOBlankingAndOrSequence(niPort_, portWidth_, blanking_, true, pos_, blankOnLow_, triggerTerminal_);
      }
      if (err != DEVICE_OK)
         sequenceRunning_ = false;
      return err;
   }

   else if (eAct == MM::StopSequence)
   {
      sequenceRunning_ = false;
      int err = GetHub()->StopDOBlankingAndSequence(portWidth_);
      if (err != DEVICE_OK)
         return err;
      if (blanking_)
         return GetHub()->StartDOBlankingAndOrSequence(niPort_, portWidth_, blanking_, false, pos_, blankOnLow_, triggerTerminal_);
      else
         return SetState(pos_);
   }

   return DEVICE_OK;
}


int DigitalOutputPort::OnTTL(MM::PropertyBase* pProp, MM::ActionType eActEx, long ttlNr)
{
   if (eActEx == MM::BeforeGet)
   {
      long state = (pos_ >> ttlNr) & 1;
      pProp->Set(state);
   }
   else if (eActEx == MM::AfterSet)
   {
      long prop;
      pProp->Get(prop);
      long pos = pos_;

      if (prop)
      {
         pos |= 1 << ttlNr; //set desired bit
      }
      else {
         pos &= ~(1 << ttlNr); // clear the second lowest bit
      }
      std::ostringstream os;
      os << pos;
      return SetProperty(MM::g_Keyword_State, os.str().c_str());
   }
   return DEVICE_OK;
}


int DigitalOutputPort::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(blanking_ ? g_On : g_Off);
   }
   else if (eAct == MM::AfterSet)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      std::string response;
      pProp->Get(response);
      bool blanking = response == g_On;
      if (blanking_ != blanking)
      {
         // do the thing in the hub
         if (blanking)
         {
            int ret = GetHub()->StartDOBlankingAndOrSequence(niPort_, portWidth_, true, false, pos_, blankOnLow_, triggerTerminal_);
            if (ret != DEVICE_OK)
               return ret;
            blanking_ = blanking;
         }
         else
         {
            int ret = GetHub()->StopDOBlankingAndSequence(portWidth_);
            if (ret != DEVICE_OK)
               return ret;
            blanking_ = blanking;
            ret =  SetState(pos_); 
            if (ret != DEVICE_OK)
               return ret;            
         }

      }
   }
   return DEVICE_OK;
}

int DigitalOutputPort::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string response = blankOnLow_ ? g_Low : g_High;
      pProp->Set(response.c_str());
   }
   else if (eAct == MM::AfterSet)
   {

      std::string response;
      pProp->Get(response);
      bool blankOnLow = response == g_Low ? true : false;
      if (blankOnLow_ != blankOnLow)
      {
         blankOnLow_ = blankOnLow;
         if (blanking_)
         {
            int ret  = GetHub()->StopDOBlankingAndSequence(portWidth_);
            if (ret != DEVICE_OK)
               return ret;
            return GetHub()->StartDOBlankingAndOrSequence(niPort_, portWidth_, true, false, pos_, blankOnLow_, triggerTerminal_);
         }            
      }
   }
   return DEVICE_OK;
}


int DigitalOutputPort::OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(neverSequenceable_ ? g_Never : g_UseHubSetting);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string s;
      pProp->Get(s);
      neverSequenceable_ = (s == g_Never);
   }
   return DEVICE_OK;
}

int DigitalOutputPort::OnNrOfStateSliders(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(nrOfStateSliders_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(nrOfStateSliders_);
   }
   return DEVICE_OK;
}


int DigitalOutputPort::StopTask()
{
   if (!task_)
      return DEVICE_OK;

   int32 nierr = DAQmxClearTask(task_);
   if (nierr != 0)
      return TranslateNIError(nierr);
   task_ = 0;
   LogMessage("Stopped task", true);

   return DEVICE_OK;
}


int DigitalOutputPort::SetState(long state)
{
   if (sequenceRunning_)
      return ERR_SEQUENCE_RUNNING;

   return GetHub()->SetDOPortState(niPort_, portWidth_, state);
}