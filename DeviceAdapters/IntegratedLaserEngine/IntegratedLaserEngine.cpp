///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#ifdef WIN32
#include <windows.h>
#endif

#include "../../MMDevice/MMDevice.h"
#include "boost/lexical_cast.hpp"
#include "ALC_REV.h"
#include "IntegratedLaserEngine.h"
#include "ILEWrapper.h"


#ifndef _isnan
#ifdef __GNUC__
#include <cmath>
using std::isnan;
#elif _MSC_VER  // MSVC.
#include <float.h>
#define isnan _isnan
#endif
#endif


// Controller.
const char* g_ControllerName = "IntegratedLaserEngine";
const char* g_DeviceList = "Device";
const char* g_Keyword_PowerSetpoint = "PowerSetpoint";
const char* g_Keyword_PowerReadback = "PowerReadback";
const char* g_Keyword_Enable = "Enable";
const char* g_Keyword_EnableOn = "On";
const char* g_Keyword_EnableOff = "Off";
const char* g_Keyword_EnableTTL = "External TTL";
const char* g_Keyword_SaveLifetime = "SaveLifetime";
const char* g_Keyword_SaveLifetimeOn = "Standby";
const char* g_Keyword_SaveLifetimeOff = "PowerOn";

const char * carriage_return = "\r";
const char * line_feed = "\n";

/** This instance is shared between the ALC and the Piezo stage. */
static CILEWrapper* pImplInstance_s;
MMThreadLock ImplLock_s;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_ControllerName, MM::ShutterDevice, "AndorLaserCombiner");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if ( (strcmp(deviceName, g_ControllerName) == 0) )
   {
      // create Controller
      CIntegratedLaserEngine* pAndorLaserCombiner = new CIntegratedLaserEngine(g_ControllerName);
      return pAndorLaserCombiner;
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// Controller implementation.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

CIntegratedLaserEngine::CIntegratedLaserEngine( const char* name ) :
  initialized_( false ),
  name_( name ),
  busy_( false ),
  error_( DEVICE_OK ),
  changedTime_( 0.0 ),
  ILEWrapper_( nullptr ),
  nLasers_( 0 ),
  openRequest_( false ),
  laserPort_( 0 ),
  ILEDevice_( nullptr )
{
  for ( int il = 0; il < MaxLasers + 1; ++il )
   {
      powerSetPoint_[il] = 0;
      isLinear_[il] = false;
      enable_[il] = g_Keyword_EnableOn;
   }

   assert(strlen(name) < (unsigned int) MM::MaxStrLength);

   InitializeDefaultErrorMessages();

   // Create pre-initialization properties:
   // -------------------------------------

   // Name
   CreateStringProperty(MM::g_Keyword_Name, name_.c_str(), true);

   // Description
   CreateStringProperty(MM::g_Keyword_Description, "Integrated Laser Engine", true);

   // Devices
   {
     MMThreadGuard G( ImplLock_s );
     if ( pImplInstance_s == nullptr )
     {
       try
       {
         ILEWrapper_ = new CILEWrapper();
       }
       catch ( std::exception &e )
       {
         LogMessage( e.what() );
         throw e;
       }
       CDeviceUtils::SleepMs( 100 );
       pImplInstance_s = ILEWrapper_;
     }
     else
     {
       ILEWrapper_ = pImplInstance_s;
     }
   }
   ILEWrapper_->GetListOfDevices( DeviceList_ );
   std::string vInitialDevice = "Undefined";
   if ( !DeviceList_.empty() )
   {
     vInitialDevice = DeviceList_.begin()->first;
   }
   CPropertyAction* pAct = new CPropertyAction( this, &CIntegratedLaserEngine::OnDeviceChange );
   CreateStringProperty( g_DeviceList, vInitialDevice.c_str(), false, pAct, true );
   std::vector<std::string> vDevices;
   CILEWrapper::TDeviceList::const_iterator vDeviceIt = DeviceList_.begin();
   while ( vDeviceIt != DeviceList_.end() )
   {
     vDevices.push_back( vDeviceIt->first );
     ++vDeviceIt;
   }
   SetAllowedValues( g_DeviceList, vDevices );


   EnableDelay(); // Signals that the delay setting will be used
   UpdateStatus();
   LogMessage(("AndorLaserCombiner ctor OK, " + std::string(name)).c_str(), true);
}

CIntegratedLaserEngine::~CIntegratedLaserEngine()
{
   Shutdown();
   MMThreadGuard g(ImplLock_s);

   // the implementation is destroyed only from the Combiner, not from ~PiezoStage
   delete ILEWrapper_;
   pImplInstance_s = NULL;
   LogMessage("AndorLaserCombiner dtor OK", true);
}

bool CIntegratedLaserEngine::Busy()
{
   MM::MMTime interval = GetCurrentMMTime() - changedTime_;
   MM::MMTime delay(GetDelayMs()*1000.0);
   if (interval < delay)
      return true;
   else
      return false;
}

void CIntegratedLaserEngine::GetName(char* name) const
{
   assert(name_.length() < CDeviceUtils::GetMaxStringLength());
   CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

int CIntegratedLaserEngine::Initialize()
{
   int nRet = DEVICE_OK;

   try
   {
     if ( !ILEWrapper_->CreateILE( &ILEDevice_, DeviceName_.c_str() ) )
     {
       LogMessage("CreateILE failed");
       return DEVICE_ERR;
     }

      nLasers_ = ILEWrapper_->ALC_REVLaser2_->Initialize();
      LogMessage(("in AndorLaserCombiner::Initialize, nLasers_ ="+boost::lexical_cast<std::string,int>(nLasers_)), true);
      CDeviceUtils::SleepMs(100);

      TLaserState state[10];
      memset((void*)state, 0, 10*sizeof(state[0]));

      //Andor says that lasers can take up to 90 seconds to initialize.
      MM::TimeoutMs timerout(GetCurrentMMTime(), 91000);
      int iloop  = 0;

      for (;;)
      {
         bool finishWaiting = true;
         for( int il = 1; il <=nLasers_; ++il)
         {
            if ( 0 == state[il])
            {
               ILEWrapper_->ALC_REVLaser2_->GetLaserState(il, state + il);
               switch( *(state + il))
               {
               case 0: // ALC_NOT_AVAILABLE ( 0) Laser is not Available
                  finishWaiting = false;
                  break;
               case 1: //ALC_WARM_UP ( 1) Laser Warming Up
                  LogMessage(" laser " + boost::lexical_cast<std::string, int>(il)+ " is warming up", true);
                  break;
               case 2: //ALC_READY ( 2) Laser is ready
                  LogMessage(" laser " + boost::lexical_cast<std::string, int>(il)+  " is ready ", true);
                  break;
               case 3: //ALC_INTERLOCK_ERROR ( 3) Interlock Error Detected
                  LogMessage(" laser " + boost::lexical_cast<std::string, int>(il) + " encountered interlock error ", false);
                  break;
               case 4: //ALC_POWER_ERROR ( 4) Power Error Detected
                  LogMessage(" laser " + boost::lexical_cast<std::string, int>(il) + " encountered power error ", false);
                  break;
               }
            }
         }
         if( finishWaiting)
            break;
         else
         {
            if (timerout.expired(GetCurrentMMTime()))
            {
               LogMessage(" some lasers did not respond", false);
               break;
            }
            iloop++;			
         }
         CDeviceUtils::SleepMs(100);
      }

      GenerateALCProperties();
      GenerateReadOnlyIDProperties();
   }
   catch (std::string& exs)
   {

      nRet = DEVICE_LOCALLY_DEFINED_ERROR;
      LogMessage(exs.c_str());
      SetErrorText(DEVICE_LOCALLY_DEFINED_ERROR,exs.c_str());
      //CodeUtility::DebugOutput(exs.c_str());
      return nRet;
   }

   initialized_ = true;
   return HandleErrors();
}


/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

void CIntegratedLaserEngine::GenerateALCProperties()
{
   std::string powerName;
   CPropertyActionEx* pAct; 
   std::ostringstream buildname;
   std::ostringstream stmp;

   // 1 based index for the lasers.
   for( int il = 1; il < nLasers_+1; ++il)
   {
      buildname.str("");
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnPowerSetpoint, il);
      buildname << g_Keyword_PowerSetpoint << Wavelength(il);
      CreateProperty(buildname.str().c_str(), "0", MM::Float, false, pAct);

      float fullScale = 10.00;
      // Set the limits as interrogated from the laser controller.
      LogMessage("Range for " + buildname.str()+"= [0," + boost::lexical_cast<std::string,float>(fullScale) + "]", true);
      SetPropertyLimits(buildname.str().c_str(), 0, fullScale);  // Volts.

      buildname.str("");
      buildname << "MaximumLaserPower" << Wavelength(il);
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnMaximumLaserPower, il );
      stmp << fullScale;
      CreateProperty(buildname.str().c_str(), stmp.str().c_str(), MM::Integer, true, pAct);

      // Readbacks.
      buildname.str("");
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnPowerReadback, il);
      buildname <<  g_Keyword_PowerReadback << Wavelength(il);
      CreateProperty(buildname.str().c_str(), "0", MM::Float, true, pAct);

      // 'States'.
      buildname.str("");
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnLaserState, il);
      buildname <<  "LaserState" << Wavelength(il);
      CreateProperty(buildname.str().c_str(), "0", MM::Integer, true, pAct);

      // Enable.
      buildname.str("");
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnEnable, il);
      buildname <<  g_Keyword_Enable << Wavelength(il);
      enableStates_[il].clear();
      enableStates_[il].push_back(g_Keyword_EnableOn);
      enableStates_[il].push_back(g_Keyword_EnableOff);
      if (AllowsExternalTTL(il))
         enableStates_[il].push_back(g_Keyword_EnableTTL);
      CreateProperty(buildname.str().c_str(), enableStates_[il][0].c_str(), MM::String, false, pAct);
      SetAllowedValues(buildname.str().c_str(),  enableStates_[il]);

      // Save laser lifetime.
      buildname.str("");
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnSaveLifetime, il);
      buildname <<  g_Keyword_SaveLifetime << Wavelength(il);
      savelifetimeStates_[il].clear();
      savelifetimeStates_[il].push_back(g_Keyword_SaveLifetimeOn);
      savelifetimeStates_[il].push_back(g_Keyword_SaveLifetimeOff);
      CreateProperty(buildname.str().c_str(), savelifetimeStates_[il][0].c_str(), MM::String, false, pAct);
      SetAllowedValues(buildname.str().c_str(), savelifetimeStates_[il]);
   }

   CPropertyAction* pA = new CPropertyAction(this, &CIntegratedLaserEngine::OnNLasers);
   CreateProperty("NLasers",  (boost::lexical_cast<std::string,int>(nLasers_)).c_str(),  MM::Integer, true, pA);
}


void CIntegratedLaserEngine::GenerateReadOnlyIDProperties()
{
   CPropertyActionEx* pAct; 
   std::ostringstream buildname;
   // 1 based index
   for( int il = 1; il < nLasers_+1; ++il)
   {
      buildname.str("");
      buildname << "Hours"  << Wavelength(il);
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnHours, il);
      CreateProperty(buildname.str().c_str(), "", MM::String, true, pAct);

      buildname.str("");
      buildname << "IsLinear"  << Wavelength(il);
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnIsLinear, il);
      CreateProperty(buildname.str().c_str(), "", MM::String, true, pAct);

      buildname.str("");
      buildname << "Wavelength"  << il;
      pAct = new CPropertyActionEx(this, &CIntegratedLaserEngine::OnWaveLength, il);
      CreateProperty(buildname.str().c_str(), "", MM::Float, true, pAct);
   }
}

int CIntegratedLaserEngine::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
      ILEWrapper_->DeleteILE( ILEDevice_ );
   }
   return HandleErrors();
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CIntegratedLaserEngine::OnDeviceChange( MM::PropertyBase* Prop, MM::ActionType eAct )
{
  if ( eAct == MM::BeforeGet )
  {
    Prop->Set( DeviceName_.c_str() );    
  }
  else if ( eAct == MM::AfterSet )
  {
    Prop->Get( DeviceName_ );
  }
  return HandleErrors();
}

/**
 * Current laser head power output.
 * <p>
 * Output of 0, could be due to laser being put in Standby using
 * SaveLifetime, or a fault with the laser head.  If power is more than
 * a few percent lower than MaximumLaserPower, it also indicates a
 * faulty laser head, but some lasers can take up to 5 minutes to warm
 * up (most warm up in 2 minutes).
 *
 * @see OnMaximumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
 */
int CIntegratedLaserEngine::OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      double v = PowerReadback((int)il);
      LogMessage(" PowerReadback" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,double>(v), true);
      pProp->Set(v);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * AOTF intensity setting.  Actual power output may or may not be
 * linear.
 *
 * @see OnIsLinear(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
 */
int CIntegratedLaserEngine::OnPowerSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct, long  il)
{
   double powerSetpoint;
   if (eAct == MM::BeforeGet)
   {
      powerSetpoint = (double)PowerSetpoint(il);
      LogMessage("from equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,double>(powerSetpoint), true);
      pProp->Set(powerSetpoint);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(powerSetpoint);
      LogMessage("to equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,double>(powerSetpoint), true);
      PowerSetpoint( il, static_cast<float>(powerSetpoint));
      if( openRequest_)
         SetOpen();

      //pProp->Set(achievedSetpoint);  ---- for quantization....
   }
   return HandleErrors();
}


/**
 * Laser bulb hours.
 * <p>
 * Indicates laser expires life to plan warranty contracts.
 * 
 * @see OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
 */
int CIntegratedLaserEngine::OnHours(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      int wval = 0;
      ILEWrapper_->ALC_REVLaser2_->GetLaserHours(il, &wval);
      LogMessage("Hours" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,int>(wval), true);
      pProp->Set((long)wval);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * Reads whether linear correction algorithm is being applied to AOTF
 * by PowerSetpoint, otherwise AOTF output is sigmoid.
 * <p>
 * Requires firmware 2.
 *
 * @see OnPowerSetpoint(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
 */
int CIntegratedLaserEngine::OnIsLinear(MM::PropertyBase* pProp, MM::ActionType eAct, long il) 
{

   if (eAct == MM::BeforeGet)
   {
      int v;
      ILEWrapper_->ALC_REVLaser2_->IsLaserOutputLinearised(il, &v);
      isLinear_[il] = (v == 1);
      long lv = static_cast<long>(v);
      LogMessage("IsLinear" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,int>(lv), true);
      pProp->Set((long)lv);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * Laser rated operating power in milli-Watts.
 *
 * @see OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
 */
int CIntegratedLaserEngine::OnMaximumLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      int val = PowerFullScale(il);
      LogMessage("PowerFullScale" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,int>(val), true);
      pProp->Set((long)val);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * Wavelength of laser line.
 */
int CIntegratedLaserEngine::OnWaveLength(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      int val = Wavelength(il);
      LogMessage("Wavelength" + boost::lexical_cast<std::string, long>(il) + "  = " + boost::lexical_cast<std::string,int>(val), true);
      pProp->Set((long)val);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * Laser state.
 */
int CIntegratedLaserEngine::OnLaserState(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      TLaserState v;
      ILEWrapper_->ALC_REVLaser2_->GetLaserState(il,&v);
      long lv = static_cast<long>(v);
      LogMessage(" LaserState" + boost::lexical_cast<std::string, long>(Wavelength(il)) + "  = " + boost::lexical_cast<std::string,long>(lv), true);
      pProp->Set(lv);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }
   return HandleErrors();
}


/**
 * Allows lasers to be put in standby mode to preserve lifetime hours.
 * Since coming out of standby can take a few seconds, best practise
 * is leave this property "global" by not including it in channels.
 * <p>
 * Requires firmware 2.
 */
int CIntegratedLaserEngine::OnSaveLifetime(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   // The SDK calls the laser standby feature "Enable".  Don't get
   // confused with the DeviceAdapter "enable" property, which is
   // just a logical shutter.
   if (eAct == MM::BeforeGet)
   {
      int v;
      ILEWrapper_->ALC_REVLaser2_->IsEnabled(il, &v);
      std::string savelifetime;
      if (v == 1)
      {
         // True value of "Enabled" corresponds to lifetime drain.
         // Therefore, lifetime saving is off.
         savelifetime_[il] = g_Keyword_SaveLifetimeOff;
      }
      else
         savelifetime_[il] = g_Keyword_SaveLifetimeOn;
      LogMessage("SaveLifetime" + boost::lexical_cast<std::string, long>(Wavelength(il)) + " = " + savelifetime_[il], true);
      pProp->Set(savelifetime_[il].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string savelifetime;
      pProp->Get(savelifetime);
      if( savelifetime_[il].compare(savelifetime) != 0 )
      {
         if (savelifetime == g_Keyword_SaveLifetimeOff)
            ILEWrapper_->ALC_REVLaser2_->Enable(il);
         else
         {
            ILEWrapper_->ALC_REVLaser2_->Disable(il);
            // SDK Bug: Disable returns true even if not supported.
            // So one has to confirm if it worked by checking
            // IsEnabled().
            int v;
            ILEWrapper_->ALC_REVLaser2_->IsEnabled(il, &v);
            if (v == 1)
               error_ = DEVICE_INVALID_PROPERTY_VALUE;
         }
         if (error_ == DEVICE_OK)
            savelifetime_[il] = savelifetime;
         LogMessage("SaveLifetime" + boost::lexical_cast<std::string, long>(Wavelength(il)) + " = " + savelifetime_[il].c_str(), true);
      }
   }
   return HandleErrors();
}


/**
 * Logical shutter to allow selection of laser line.  It can also set
 * the laser to TTL mode, if the laser supports it.
 * <p>
 * TTL mode requires firmware 2.
 */
int CIntegratedLaserEngine::OnEnable(MM::PropertyBase* pProp, MM::ActionType eAct, long il)
{
   if (eAct == MM::BeforeGet)
   {
      // Not calling GetControlMode() from ALC SDK, since it may slow
      // down acquisition while switching channels.
      pProp->Set(enable_[il].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string enable;
      pProp->Get(enable);
      if( enable_[il].compare(enable) != 0 )
      {
         // Update the laser control mode if we are changing to, or
         // from External TTL mode.
         if ( enable.compare(g_Keyword_EnableTTL) == 0 )
            ILEWrapper_->ALC_REVLaser2_->SetControlMode(il, TTL_PULSED);
         else if ( enable_[il].compare(g_Keyword_EnableTTL) == 0 )
            ILEWrapper_->ALC_REVLaser2_->SetControlMode(il, CW);

         enable_[il] = enable;
         LogMessage("Enable" + boost::lexical_cast<std::string, long>(Wavelength(il)) + " = " + enable_[il], true);
         if( openRequest_)
            SetOpen();
      }
   }
   return HandleErrors();
}


/**
 * Number of lasers available.
 */
int CIntegratedLaserEngine::OnNLasers(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      long v = nLasers_;
      pProp->Set(v);
   }
   else if (eAct == MM::AfterSet)
   {
      // This is a read-only property!
   }   
   return HandleErrors();
}


int CIntegratedLaserEngine::HandleErrors()
{
   int lastError = error_;
   error_ = DEVICE_OK;
   return lastError;
}


//********************
// Shutter API
//********************

int CIntegratedLaserEngine::SetOpen(bool open)
{
   for( int il = 1; il <= this->nLasers_; ++il)
   {
      if(open)
      {
         double fullScale = 10.00; /* Volts instead of milliWatts, and  double instead of int */
         bool onn = ( 0 < PowerSetpoint(il))  && (enable_[il].compare(g_Keyword_EnableOff) != 0);
         double percentScale = 0.;
         if( onn)
            percentScale = 100.*PowerSetpoint(il)/fullScale;

         if( 100. < percentScale )
            percentScale = 100.;
         LogMessage("SetLas" + boost::lexical_cast<std::string, long>(il) + "  = " + boost::lexical_cast<std::string,double>(percentScale) + "(" + boost::lexical_cast<std::string,bool>(onn)+")" , true);

         TLaserState ttmp;
         ILEWrapper_->ALC_REVLaser2_->GetLaserState(il, &ttmp);
         if( onn && ( 2 != ttmp))
         {
            std::string messg = "Laser # " + boost::lexical_cast<std::string,int>(il) + " is not ready!";
            // laser is not ready!
            LogMessage(messg.c_str(), false);
            // GetCoreCallback()->PostError(std::make_pair<int,std::string>(DEVICE_ERR,messg));
         }

         if( ALC_NOT_AVAILABLE < ttmp)
         {
            LogMessage("setting Laser " + boost::lexical_cast<std::string,int>(Wavelength(il)) + " to " + boost::lexical_cast<std::string, double>(percentScale) + "% full scale", true);
            ILEWrapper_->ALC_REVLaser2_->SetLas_I( il,percentScale, onn );
         }

		}
      LogMessage("set shutter " + boost::lexical_cast<std::string, bool>(open), true);
      bool succ = ILEWrapper_->ALC_REVLaser2_->SetLas_Shutter(open);
      if( !succ)
         LogMessage("set shutter " + boost::lexical_cast<std::string, bool>(open) + " failed", false);
   }

   openRequest_ = open;

   return DEVICE_OK;
}


int CIntegratedLaserEngine::GetOpen(bool& open)
{
   // todo check that all requested lasers are 'ready'.
   open = openRequest_ ; // && Ready();
   return DEVICE_OK;
}

/**
 * ON for deltaT milliseconds.  Other implementations of Shutter don't
 * implement this.  Is this perhaps because this blocking call is not
 * appropriate?
 */
int CIntegratedLaserEngine::Fire(double deltaT)
{
   SetOpen(true);
   CDeviceUtils::SleepMs((long)(deltaT+.5));
   SetOpen(false);
   return HandleErrors();
}


int CIntegratedLaserEngine::Wavelength(const int laserIndex_a)
{
   int wval = 0;
   ILEWrapper_->ALC_REVLaser2_->GetWavelength(laserIndex_a, &wval);
   return wval;
}

int CIntegratedLaserEngine::PowerFullScale(const int laserIndex_a)
{
   int val = 0;
   ILEWrapper_->ALC_REVLaser2_->GetPower(laserIndex_a, &val);
   return val;
}

float CIntegratedLaserEngine::PowerReadback(const int laserIndex_a)
{
   double val = 0.;
   ILEWrapper_->ALC_REVLaser2_->GetCurrentPower(laserIndex_a, &val);
   if( isnan(val))
   {
      LogMessage("invalid PowerReadback on # " + boost::lexical_cast<std::string,int>(laserIndex_a), false);
      val = 0.;
   }
   return (float) val;
}

float CIntegratedLaserEngine::PowerSetpoint(const int laserIndex_a)
{
   return powerSetPoint_[laserIndex_a];
}

void  CIntegratedLaserEngine::PowerSetpoint(const int laserIndex_a, const float val_a)
{
   powerSetPoint_[laserIndex_a] = val_a;
}

bool CIntegratedLaserEngine::AllowsExternalTTL(const int laserIndex_a)
{
   int val = 0;
   ILEWrapper_->ALC_REVLaser2_->IsControlModeAvailable(laserIndex_a, &val);
   return (val == 1);
}

bool CIntegratedLaserEngine::Ready(const int laserIndex_a)
{
   TLaserState state = ALC_NOT_AVAILABLE;
   bool ret =	ILEWrapper_->ALC_REVLaser2_->GetLaserState(laserIndex_a, &state);
   return ret && ( ALC_READY == state);	
}
