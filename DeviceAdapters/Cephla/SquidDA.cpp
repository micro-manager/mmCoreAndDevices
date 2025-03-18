#include "Squid.h"

const char* g_DAName = "DA";
const char* g_Volts = "Volts";
const char* g_VoltRange = "Volt-Range";
const char* g_0_5V = "0-5V";
const char* g_0_2_5V = "0-2.5V";


SquidDA::SquidDA(uint8_t dacNr) :
   hub_(0),
   initialized_(false),
   busy_(false),
   volts_(0.0),
   maxV_(5.0),
   gatedVolts_(0.0),
   gateOpen_(true),
   dacNr_(dacNr)
{
   CPropertyAction* pAct = new CPropertyAction(this, &SquidDA::OnVoltRange);
   CreateStringProperty(g_VoltRange, g_0_5V, false, pAct, true);
   AddAllowedValue(g_VoltRange, g_0_2_5V);
   AddAllowedValue(g_VoltRange, g_0_5V);
}


SquidDA::~SquidDA()
{
   if (initialized_)
   {
      Shutdown();
   }
}


int SquidDA::Shutdown() {
   initialized_ = false;
   return DEVICE_OK;
}


void SquidDA::GetName(char* pszName) const
{
   std::ostringstream os;
   os << g_DAName << "_" << ((int)dacNr_ + 1);
   CDeviceUtils::CopyLimitedString(pszName, os.str().c_str());
}


int SquidDA::Initialize()
{
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   bool setGain = maxV_ == 5.0;;
   int ret = hub_->SetDacGain((uint8_t) dacNr_, setGain);
   if (ret != DEVICE_OK)
       return ret;

   CPropertyAction* pAct = new CPropertyAction(this, &SquidDA::OnVolts);
   ret = CreateFloatProperty(g_Volts, 0.0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Volts, 0.0, maxV_);

   return DEVICE_OK;
}


bool SquidDA::Busy() { return hub_->Busy(); }


int SquidDA::SetGateOpen(bool open)
{
   gateOpen_ = open;
   if (open)
   {
      SendVoltage(volts_);
   }
   else
   {
      SendVoltage(0.0);
   }
   return DEVICE_OK;
}

int SquidDA::GetGateOpen(bool& open)
{
   open = gateOpen_;
   return DEVICE_OK;
};


int SquidDA::SetSignal(double volts)
{
   volts_ = volts;
   if (gateOpen_)
   {
      return SendVoltage(volts_);
   }
   return DEVICE_OK;
}

int SquidDA::GetSignal(double& volts)
{
   volts_ = volts;
   return DEVICE_OK;
}


int SquidDA::GetLimits(double& minVolts, double& maxVolts)
{
   minVolts = 0;
   maxVolts = maxV_;
   return DEVICE_OK;
}


int SquidDA::IsDASequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }


// action interface
// ----------------
int SquidDA::OnVoltRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxV_ == 2.5 ? g_0_2_5V : g_0_5V);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string response;
      pProp->Get(response);
      maxV_ = response == g_0_2_5V ? 2.5 : 5.0;
   }
   return DEVICE_OK;
}


int SquidDA::OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(volts_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(volts_);
      return SetSignal(volts_);
   }
   return DEVICE_OK;
}


int SquidDA::SendVoltage(double volts)
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_ANALOG_WRITE_ONBOARD_DAC;
   cmd[2] = (uint8_t) dacNr_;
   uint16_t value = (uint16_t) (volts / maxV_ * 65535);
   cmd[3] = (value >> 8) & 0xff;
   cmd[4] = value & 0xff;
   return hub_->SendCommand(cmd, cmdSize);
}


