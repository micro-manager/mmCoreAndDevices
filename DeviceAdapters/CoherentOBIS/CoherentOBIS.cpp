///////////////////////////////////////////////////////////////////////////////
// FILE:          CoherentObis.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   CoherentObis controller adapter
// COPYRIGHT:     
//                MBL, Woods Hole, MA 2014
//                University of California, San Francisco, 2009 (Hoover)
//
// AUTHOR:        Forrest Collman
//                Adapted from CoherentCube driver written by Karl Hoover, UCSF
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifdef WIN32
   #include <windows.h>
#endif
#include "FixSnprintf.h"

#include "CoherentOBIS.h"

#include "MMDevice.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"

#include <algorithm>
#include <math.h>
#include <sstream>
#include <string>


// Controller
const char* g_ControllerName = "CoherentObis";
const char* g_Keyword_PowerSetpoint = "PowerSetpoint";
const char* g_Keyword_PowerReadback = "PowerReadback";

const char * carriage_return = "\r";
const char * line_feed = "\n";




///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_ControllerName, MM::ShutterDevice, "CoherentObis Laser");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_ControllerName) == 0)
   {
      // create Controller
      CoherentObis* pCoherentObis = new CoherentObis(g_ControllerName);
      return pCoherentObis;
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// Controller implementation
// ~~~~~~~~~~~~~~~~~~~~

CoherentObis::CoherentObis(const char* name) :
   initialized_(false), 
   state_(0),
   name_(name), 
   error_(0),
   changedTime_(0.0),
   queryToken_("?"),
   CDRHToken_("CDRH"),  // if this is on, laser delays 5 SEC before turning on
   CWToken_("CW"),
   TECServoToken_("T")
{
   assert(strlen(name) < (unsigned int) MM::MaxStrLength);

   InitializeDefaultErrorMessages();
   SetErrorText(ERR_DEVICE_NOT_FOUND, "No answer received.  Is the Coherent Cube connected to this serial port?");
   // create pre-initialization properties
   // ------------------------------------

   // Name
   CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);

   // Description
   CreateProperty(MM::g_Keyword_Description, "CoherentObis Laser", MM::String, true);

   // Port
   CPropertyAction* pAct = new CPropertyAction (this, &CoherentObis::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   EnableDelay(); // signals that the delay setting will be used

   // Default device index is 1 (uses SYST1/SOUR1)
   deviceIndex_ = 1;

   CPropertyAction* pActDeviceIndex = new CPropertyAction(this, &CoherentObis::OnDeviceIndex);
   CreateProperty("DeviceIndex", "1", MM::Integer, false, pActDeviceIndex, true);
   UpdateStatus();
}

CoherentObis::~CoherentObis()
{
   Shutdown();
}

bool CoherentObis::Busy()
{
   MM::MMTime interval = GetCurrentMMTime() - changedTime_;
   MM::MMTime delay(GetDelayMs()*1000.0);
   if (interval < delay)
      return true;
   else
      return false;
}

void CoherentObis::GetName(char* name) const
{
   assert(name_.length() < CDeviceUtils::GetMaxStringLength());
   CDeviceUtils::CopyLimitedString(name, name_.c_str());
}


int CoherentObis::Initialize()
{
   LogMessage("CoherentObis::Initialize()yes??");

   GeneratePowerProperties();
   GeneratePropertyState();
   GenerateReadOnlyIDProperties();
   std::stringstream msg;


   //Initialize laser??
   setLaser(getPrefix() + ":COMM:HAND","On");
   setLaser(getPrefix() + ":COMM:PROM","Off");
   msg << getPrefix() + ":ERR:CLE" ;
   Send(msg.str());

   // query laser for power limits
   this->initLimits();

   double llimit = this->minlp();
   double ulimit = this->maxlp();

   // set the limits as interrogated from the laser controller.
   SetPropertyLimits(g_Keyword_PowerSetpoint, llimit, ulimit);  // milliWatts
   
   initialized_ = true;

   return HandleErrors();
}


/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

void CoherentObis::GeneratePropertyState()
{
   CPropertyAction* pAct = new CPropertyAction (this, &CoherentObis::OnState);
   CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
   AddAllowedValue(MM::g_Keyword_State, "0");
   AddAllowedValue(MM::g_Keyword_State, "1");
}


void CoherentObis::GeneratePowerProperties()
{
   string powerName;

   // Power Setpoint
   CPropertyActionEx* pActEx = new CPropertyActionEx(this, &CoherentObis::OnPowerSetpoint, 0);
   powerName = g_Keyword_PowerSetpoint;
   CreateProperty(powerName.c_str(), "0", MM::Float, false, pActEx);

   // Power Setpoint
   pActEx = new CPropertyActionEx(this, &CoherentObis::OnPowerReadback, 0);
   powerName = g_Keyword_PowerReadback;
   CreateProperty(powerName.c_str(), "0", MM::Float, true, pActEx);
}


void CoherentObis::GenerateReadOnlyIDProperties()
{
   CPropertyAction* pAct; 
   pAct = new CPropertyAction(this, &CoherentObis::OnHeadID);
   CreateProperty("HeadID", "", MM::String, true, pAct);

   pAct = new CPropertyAction(this, &CoherentObis::OnHeadUsageHours);
   CreateProperty("Head Usage Hours", "", MM::String, true, pAct);

   pAct = new CPropertyAction(this, &CoherentObis::OnMinimumLaserPower);
   CreateProperty("Minimum Laser Power", "", MM::Float, true, pAct);
   
   pAct = new CPropertyAction(this, &CoherentObis::OnMaximumLaserPower);
   CreateProperty("Maximum Laser Power", "", MM::Float, true, pAct);

   pAct = new CPropertyAction(this, &CoherentObis::OnWaveLength);
   CreateProperty("Wavelength", "", MM::Float, true, pAct);
}

int CoherentObis::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return HandleErrors();
}

std::string CoherentObis::getPrefix() const
{
    std::ostringstream oss;
    oss << "SYST" << deviceIndex_;
    return oss.str();
}

std::string CoherentObis::getPowerPrefix() const 
{
    std::ostringstream oss;
    oss << "SOUR" << deviceIndex_;
    return oss.str();
}

std::string CoherentObis::getPowerSetpointToken() const
{
   return getPowerPrefix() + ":POW:LEV:IMM:AMPL";
}

std::string CoherentObis::getPowerReadbackToken() const 
{
   return getPowerPrefix() + ":POW:LEV:IMM:AMPL";
}

std::string CoherentObis::getLaserOnToken() const 
{
   return getPowerPrefix() + ":AM:STATE";
}

std::string CoherentObis::getHeadSerialNoToken() const
{
   return getPrefix() + ":INF:SNUM";
}

std::string CoherentObis::getHeadUsageHoursToken() const
{
   return getPrefix() + ":DIOD:HOUR";
}

std::string CoherentObis::getWavelengthToken() const
{
   return getPrefix() + ":INF:WAV";
}

std::string CoherentObis::getExternalPowerControlToken() const
{
    return getPowerPrefix() + ":POW:LEV:IMM:AMPL";
}

std::string CoherentObis::getMaxPowerToken() const
{
   return getPowerPrefix() + ":POW:LIM:HIGH";
}

std::string CoherentObis::getMinPowerToken() const
{
   return getPowerPrefix() + ":POW:LIM:LOW";
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CoherentObis::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         // revert
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }

      pProp->Get(port_);
   }

   return HandleErrors();
}

int CoherentObis::OnDeviceIndex(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) 
   {
      pProp->Set(static_cast<long>(deviceIndex_));
   } 
   else if (eAct == MM::AfterSet) 
   {
      long val;
      pProp->Get(val);
      if (val == 0 || val == 1) 
      {
         deviceIndex_ = static_cast<int>(val);
      } 
      else 
      {
         return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }

   return HandleErrors();
}


int CoherentObis::OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long /*index*/)
{

   double powerReadback;
   if (eAct == MM::BeforeGet)
   {
      GetPowerReadback(powerReadback);
      pProp->Set(powerReadback);
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}

int CoherentObis::OnPowerSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct, long  /*index*/)
{

   double powerSetpoint;
   if (eAct == MM::BeforeGet)
   {
      GetPowerSetpoint(powerSetpoint);
      pProp->Set(powerSetpoint);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(powerSetpoint);
      double achievedSetpoint;
      SetPowerSetpoint(powerSetpoint, achievedSetpoint);
      if( 0. != powerSetpoint)
      {
         double fractionError = fabs(achievedSetpoint - powerSetpoint) / powerSetpoint;
         if (( 0.05 < fractionError ) && (fractionError  < 0.10))
            pProp->Set(achievedSetpoint);
      }
   }
   return HandleErrors();
}


int CoherentObis::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      GetState(state_);
      pProp->Set(state_);
   }
   else if (eAct == MM::AfterSet)
   {
      long requestedState;
      pProp->Get(requestedState);
      SetState(requestedState);
      if (state_ != requestedState)
      {
         // error
      }
   }
   
   return HandleErrors();
}


int CoherentObis::OnHeadID(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((this->queryLaser(getHeadSerialNoToken())).c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}


int CoherentObis::OnHeadUsageHours(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string svalue = this->queryLaser(getHeadUsageHoursToken());
      double dvalue = atof(svalue.c_str());
      pProp->Set(dvalue);
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}


int CoherentObis::OnMinimumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(atof((this->queryLaser(getMinPowerToken())).c_str()));
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}

int CoherentObis::OnMaximumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(atof((this->queryLaser(getMaxPowerToken())).c_str()));
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}


int CoherentObis::OnWaveLength(MM::PropertyBase* pProp, MM::ActionType eAct /* , long */)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(atof((this->queryLaser(getWavelengthToken())).c_str()));
   }
   else if (eAct == MM::AfterSet)
   {
      // never do anything!!
   }
   return HandleErrors();
}

void CoherentObis::GetPowerReadback(double& value)
{
   string ans = this->queryLaser(getPowerReadbackToken());
   value = POWERCONVERSION*atof(ans.c_str());
}

void CoherentObis::SetPowerSetpoint(double requestedPowerSetpoint, double& achievedPowerSetpoint)
{
   std::string result;
   std::ostringstream setpointString;
   // number like 100.00
   setpointString << setprecision(6) << requestedPowerSetpoint/POWERCONVERSION;
   result = this->setLaser(getPowerSetpointToken(), setpointString.str());
   //compare quantized setpoint to requested setpoint
   // the difference can be rather large

   achievedPowerSetpoint = POWERCONVERSION*atof( result.c_str());

   // if device echos a setpoint more the 10% of full scale from requested setpoint, log a warning message
   if ( this->maxlp()/10. < fabs( achievedPowerSetpoint-POWERCONVERSION*requestedPowerSetpoint))
   {
      std::ostringstream messs;
      messs << "requested setpoint: " << requestedPowerSetpoint << " but echo setpoint is: " << achievedPowerSetpoint;
      LogMessage(messs.str().c_str());
   }
}

void CoherentObis::GetPowerSetpoint(double& value)
{
   string ans = this->queryLaser(getPowerSetpointToken());
   value = POWERCONVERSION*atof(ans.c_str());
}

void CoherentObis::SetState(long state)
{
   std::ostringstream atoken;
   if (state==1){
      atoken << "On";
   }
   else{
      atoken << "Off";
   }
   this->setLaser(getLaserOnToken(), atoken.str());
   // Set timer for the Busy signal
   changedTime_ = GetCurrentMMTime();
}

void CoherentObis::GetState(long &value)
{
   string ans = this->queryLaser(getLaserOnToken());
   std::transform(ans.begin(), ans.end(), ans.begin(), ::tolower);
   if (ans.find("on") == 0) {
      value = 1;
   }
   else if (ans.find("off") == 0) {
      value = 0;
   }
   else{
      value = 2;
   }
}

void CoherentObis::SetExternalLaserPowerControl(int value)
{
   std::ostringstream atoken;
   atoken << value;
   this->setLaser(getExternalPowerControlToken(), atoken.str());
}

void CoherentObis::GetExternalLaserPowerControl(int& value)
{
   string ans = this->queryLaser(getExternalPowerControlToken());
   value = atol(ans.c_str());
}

int CoherentObis::HandleErrors()
{
   int lastError = error_;
   error_ = 0;
   return lastError;
}



/////////////////////////////////////
//  Communications
/////////////////////////////////////


void CoherentObis::Send(string cmd)
{
   std::ostringstream messs;
   messs << "CoherentObis::Send           " << cmd;
   LogMessage( messs.str().c_str(), true);

   int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), carriage_return);
   if (ret!=DEVICE_OK)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;
}


int CoherentObis::ReceiveOneLine()
{
   buf_string_ = "";
   int ret = GetSerialAnswer(port_.c_str(), line_feed, buf_string_);
   if (ret != DEVICE_OK)
      return ret;
   std::ostringstream messs;
   messs << "CoherentObis::ReceiveOneLine " << buf_string_;
   LogMessage( messs.str().c_str(), true);

   return DEVICE_OK;
}

void CoherentObis::Purge()
{
   int ret = PurgeComPort(port_.c_str());
   if (ret!=0)
      error_ = DEVICE_SERIAL_COMMAND_FAILED;
}

//********************
// Shutter API
//********************

int CoherentObis::SetOpen(bool open)
{
   SetState((long) open);
   return HandleErrors();
}

int CoherentObis::GetOpen(bool& open)
{
   long state;
   GetState(state);
   if (state==1)
      open = true;
   else if (state==0)
      open = false;
   else
      error_ = DEVICE_UNKNOWN_POSITION;

   return HandleErrors();
}

// ON for deltaT milliseconds
// other implementations of Shutter don't implement this
// is this perhaps because this blocking call is not appropriate
int CoherentObis::Fire(double deltaT)
{
   SetOpen(true);
   CDeviceUtils::SleepMs((long)(deltaT+.5));
   SetOpen(false);
   return HandleErrors();
}
