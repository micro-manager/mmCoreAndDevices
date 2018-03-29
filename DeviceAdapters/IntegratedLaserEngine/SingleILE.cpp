///////////////////////////////////////////////////////////////////////////////
// FILE:          SingleILE.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#include "ALC_REV.h"
#include "SingleILE.h"
#include "Ports.h"
#include "ActiveBlanking.h"
#include "LowPowerMode.h"


// Properties
const char* const CSingleILE::g_DeviceName = "Andor ILE";
const char* const CSingleILE::g_DeviceDescription = "Integrated Laser Engine";

CSingleILE::CSingleILE() :
  CIntegratedLaserEngine( g_DeviceDescription, 1 ),
  Ports_( nullptr ),
  ActiveBlanking_( nullptr ),
  LowPowerMode_( nullptr )
{
  LogMessage( std::string( g_DeviceName ) + " ctor OK", true );
}

CSingleILE::~CSingleILE()
{
  LogMessage( std::string( g_DeviceName ) + " dtor OK", true );
}

std::string CSingleILE::GetDeviceName() const
{
  return g_DeviceName;
}

int CSingleILE::Shutdown()
{
  delete LowPowerMode_;
  LowPowerMode_ = nullptr;
  delete ActiveBlanking_;
  ActiveBlanking_ = nullptr;
  delete Ports_;
  Ports_ = nullptr;
  return CIntegratedLaserEngine::Shutdown();
}

bool CSingleILE::CreateILE()
{
  bool vRet = false;
  if ( DevicesNames_.size() > 0 )
  {
    LogMessage( "Creating ILE for device " + DevicesNames_[0], true );
    vRet = ILEWrapper_->CreateILE( &ILEDevice_, DevicesNames_[0].c_str() );
  }
  return vRet;
}

void CSingleILE::DeleteILE()
{
  ILEWrapper_->DeleteILE( ILEDevice_ );
  ILEDevice_ = nullptr;
}

void CSingleILE::DisconnectILEInterfaces()
{
  if ( LowPowerMode_ )
  {
    LowPowerMode_->UpdateILEInterface( nullptr );
  }
  if ( ActiveBlanking_ )
  {
    ActiveBlanking_->UpdateILEInterface( nullptr );
  }
  if ( Ports_ )
  {
    Ports_->UpdateILEInterface( nullptr );
  }
}

int CSingleILE::ReconnectILEInterfaces()
{
  IALC_REV_Port* vPortInterface = ILEDevice_->GetPortInterface();
  IALC_REV_ILEActiveBlankingManagement* vActiveBlankingInterface = ILEWrapper_->GetILEActiveBlankingManagementInterface( ILEDevice_ );
  IALC_REV_ILEPowerManagement* vLowPowerModeInterface = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  int vRet = DEVICE_OK;
  if ( Ports_ != nullptr )
  {
    if ( vPortInterface != nullptr )
    {
      vRet = Ports_->UpdateILEInterface( vPortInterface );
    }
    else
    {
      LogMessage( "Pointer to the Port inteface invalid" );
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  if ( vRet == DEVICE_OK && ActiveBlanking_ != nullptr )
  {
    if ( vActiveBlankingInterface != nullptr )
    {
      vRet = ActiveBlanking_->UpdateILEInterface( vActiveBlankingInterface );
    }
    else
    {
      LogMessage( "Pointer to the Active Blanking inteface invalid" );
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  if ( vRet == DEVICE_OK && LowPowerMode_ != nullptr )
  {
    if ( vLowPowerModeInterface != nullptr )
    {
      vRet = LowPowerMode_->UpdateILEInterface( vLowPowerModeInterface );
    }
    else
    {
      LogMessage( "Pointer to the Low Power Mode inteface invalid" );
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  return vRet;
}

int CSingleILE::InitializePorts()
{
  IALC_REV_Port* vPortInterface = ILEDevice_->GetPortInterface();
  if ( vPortInterface != nullptr )
  {
    try
    {
      Ports_ = new CPorts( vPortInterface, this );
    }
    catch ( std::exception& vException )
    {
      std::string vMessage( "Error loading the Ports. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_PORTS_INIT;
    }
  }
  else
  {
    LogMessage( "Port interface pointer invalid" );
  }
  return DEVICE_OK;
}

int CSingleILE::InitializeActiveBlanking()
{
  IALC_REV_ILEActiveBlankingManagement* vActiveBlanking = ILEWrapper_->GetILEActiveBlankingManagementInterface( ILEDevice_ );
  if ( vActiveBlanking != nullptr )
  {
    bool vActiveBlankingPresent = false;
    if ( !vActiveBlanking->IsActiveBlankingManagementPresent( &vActiveBlankingPresent ) )
    {
      LogMessage( "Active Blanking IsActiveBlankingManagementPresent failed" );
      return ERR_ACTIVEBLANKING_INIT;
    }
    if ( vActiveBlankingPresent )
    {
      try
      {
        ActiveBlanking_ = new CActiveBlanking( vActiveBlanking, this );
      }
      catch ( std::exception& vException )
      {
        std::string vMessage( "Error loading Active Blanking. Caught Exception with message: " );
        vMessage += vException.what();
        LogMessage( vMessage );
        return ERR_ACTIVEBLANKING_INIT;
      }
    }
    else
    {
      LogMessage( "Active Blanking not present", true );
    }
  }
  else
  {
    LogMessage( "Active Blanking interface pointer invalid" );
  }
  return DEVICE_OK;
}

int CSingleILE::InitializeLowPowerMode()
{
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  if ( vLowPowerMode != nullptr )
  {
    bool vLowPowerModePresent = false;
    if ( !vLowPowerMode->IsLowPowerPresent( &vLowPowerModePresent ) )
    {
      LogMessage( "ILE Power IsLowPowerPresent failed" );
      return ERR_LOWPOWERMODE_INIT;
    }
    if ( vLowPowerModePresent )
    {
      try
      {
        LowPowerMode_ = new CLowPowerMode( vLowPowerMode, this );
      }
      catch ( std::exception& vException )
      {
        std::string vMessage( "Error loading Low Power mode. Caught Exception with message: " );
        vMessage += vException.what();
        LogMessage( vMessage );
        return ERR_LOWPOWERMODE_INIT;
      }
    }
    else
    {
      LogMessage( "Low Power Mode not present", true );
    }
  }
  else
  {
    LogMessage( "ILE Power interface pointer invalid" );
  }
  return DEVICE_OK;
}
