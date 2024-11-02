#include "Squid.h"

const char* g_DAName = "DA";
const char* g_Volts = "Volts";


SquidDA::SquidDA() :
   hub_(0),
   initialized_(false),
   busy_(false),
   volts_(0.0),
   maxV_(5.0),
   gatedVolts_(0.0),
   gateOpen_(true),
   dacNr_(7)
{
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
   CDeviceUtils::CopyLimitedString(pszName, g_DAName);
}


int SquidDA::Initialize()
{
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   CPropertyAction* pAct = new CPropertyAction(this, &SquidDA::OnVolts);
   int ret = CreateFloatProperty(g_Volts, 0.0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Volts, 0.0, 5.0);

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
   maxVolts = 5;
   return DEVICE_OK;
}


int SquidDA::IsDASequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

// action interface
// ----------------
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


int SquidDA::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Get(dacNr_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set(dacNr_);
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


