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
#include "NDFilters.h"
#include "LowPowerMode.h"


// Properties
const char* const CSingleILE::g_DeviceName = "Andor ILE";
const char* const CSingleILE::g_DeviceDescription = "Integrated Laser Engine";

CSingleILE::CSingleILE() :
  CIntegratedLaserEngine( g_DeviceDescription, 1 ),
  Ports_( nullptr ),
  ActiveBlanking_( nullptr ),
  NDFilters_( nullptr ),
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
  LogMessage( "Single ILE shutdown", true );

  if ( NDFilters_ )
  {
    delete NDFilters_;
    NDFilters_ = nullptr;
  }
  if ( LowPowerMode_ )
  {
    delete LowPowerMode_;
    LowPowerMode_ = nullptr;
  }
  if ( ActiveBlanking_ )
  {
    delete ActiveBlanking_;
    ActiveBlanking_ = nullptr;
  }
  if ( Ports_ )
  {
    delete Ports_;
    Ports_ = nullptr;
  }

  LogMessage( "Single ILE shutdown done", true );
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
  if ( NDFilters_ != nullptr )
  {
    NDFilters_->UpdateILEInterface( nullptr );
  }
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
  if ( vRet == DEVICE_OK )
  {
    vRet = ReconnectNDFilters();
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

int CSingleILE::ReconnectNDFilters()
{
  int vRet = DEVICE_OK;
  if ( NDFilters_ != nullptr )
  {
    IALC_REV_ILEPowerManagement2* vPowerManagement2 = ILEWrapper_->GetILEPowerManagement2Interface( ILEDevice_ );
    if ( vPowerManagement2 != nullptr )
    {
      vRet = NDFilters_->UpdateILEInterface( vPowerManagement2 );
    }
    else
    {
      LogMessage( "ILE Power Management 2 interface pointer invalid" );
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

int CSingleILE::InitializeNDFilters()
{
  IALC_REV_ILEPowerManagement2* vPowerManagement2 = ILEWrapper_->GetILEPowerManagement2Interface( ILEDevice_ );
  if ( vPowerManagement2 != nullptr )
  {
    bool vLowPowerModePresent = false;
    if ( !vPowerManagement2->IsLowPowerPresent( &vLowPowerModePresent ) )
    {
      LogMessage( "ILE IsLowPowerPresent failed" );
      return ERR_NDFILTERS_INIT;
    }

    if ( vLowPowerModePresent )
    {
      int vNumLevels;
      if ( !vPowerManagement2->GetNumberOfLowPowerLevels( &vNumLevels ) )
      {
        LogMessage( "ILE GetNumberOfLowPowerLevels failed" );
        return ERR_NDFILTERS_INIT;
      }

      if ( vNumLevels > 1 )
      {
        try
        {
          NDFilters_ = new CNDFilters( vPowerManagement2, this );
        }
        catch ( std::exception& vException )
        {
          std::string vMessage( "Error loading ND Filters. Caught Exception with message: " );
          vMessage += vException.what();
          LogMessage( vMessage );
          return ERR_NDFILTERS_INIT;
        }
      }
      else
      {
        LogMessage( "Low Power only has 1 level, ND Filters is not present on this device, reverting to old style Low Power UI", true );
        return InitializeLowPowerMode();
      }
    }
    else
    {
      LogMessage( "Low Power (ND Filters) not present", true );
    }
  }
  else
  {
    LogMessage( "ILE Power Management 2 interface pointer invalid, trying to revert to old ILE Power Management", true );
    // Since ND Filters are not available, trying old fashion Low Power mode
    return InitializeLowPowerMode();
  }
  return DEVICE_OK;
}

void CSingleILE::CheckAndUpdateLowPowerMode()
{
  if ( NDFilters_ != nullptr )
  {
    NDFilters_->CheckAndUpdate();
  }

  if ( LowPowerMode_ != nullptr )
  {
    LowPowerMode_->CheckAndUpdate();
  }
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
    LogMessage( "ILE Power Management interface pointer invalid" );
  }
  return DEVICE_OK;
}
