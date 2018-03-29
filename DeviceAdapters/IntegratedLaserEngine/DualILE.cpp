///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILE.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
#include "DualILE.h"
#include "PortsConfiguration.h"
#include "DualILEPorts.h"
#include "DualILEActiveBlanking.h"
#include "DualILELowPowerMode.h"


// Properties
const char* const CDualILE::g_DualDeviceName = "Andor Dual ILE";
const char* const CDualILE::g_Dual700DeviceName = "Andor Dual ILE 700";
const char* const CDualILE::g_DualDeviceDescription = "Dual Integrated Laser Engine";
const char* const CDualILE::g_Dual700DeviceDescription = "Dual Integrated Laser Engine 700";


CDualILE::CDualILE( bool ILE700 ) :
  CIntegratedLaserEngine( ( ILE700 ? g_Dual700DeviceDescription : g_DualDeviceDescription ), 2 ),
  ILE700_( ILE700 ),
  Ports_( nullptr ),
  PortsConfiguration_( nullptr ),
  ActiveBlanking_( nullptr ),
  LowPowerMode_( nullptr )
{
  SetErrorText( ERR_DUALPORTS_PORTCONFIGCORRUPTED, "Changing port failed. Port configuration is corrupted" );
  SetErrorText( ERR_DUALPORTS_PORTCHANGEFAIL, "Changing port failed." );
  SetErrorText( ERR_DUALILE_GETINTERFACE, "Dual ILE GetInterface failed" );
  SetErrorText( ERR_LOWPOWERPRESENT, "Failed to retrieve low power presence" );
  LogMessage( std::string( g_DualDeviceName ) + " ctor OK", true );
}

CDualILE::~CDualILE()
{
  delete PortsConfiguration_;
  LogMessage( std::string( g_DualDeviceName ) + " dtor OK", true );
}

std::string CDualILE::GetDeviceName() const
{
  std::string vName = g_DualDeviceName;
  if ( ILE700_ )
  {
    vName = g_Dual700DeviceName;
  }
  return vName;
}

int CDualILE::Shutdown()
{
  delete LowPowerMode_;
  LowPowerMode_ = nullptr;
  delete ActiveBlanking_;
  ActiveBlanking_ = nullptr;
  delete Ports_;
  Ports_ = nullptr;
  return CIntegratedLaserEngine::Shutdown();
}

bool CDualILE::CreateILE()
{
  bool vRet = false;
  if ( DevicesNames_.size() > 1 )
  {
    LogMessage( "Creating Dual ILE for devices " + DevicesNames_[0] + " and " + DevicesNames_[1], true );
    vRet = ILEWrapper_->CreateDualILE( &ILEDevice_, DevicesNames_[0].c_str(), DevicesNames_[1].c_str(), ILE700_ );
  }
  return vRet;
}

void CDualILE::DeleteILE()
{
  ILEWrapper_->DeleteDualILE( ILEDevice_ );
  ILEDevice_ = nullptr;
}

void CDualILE::DisconnectILEInterfaces()
{
  if ( LowPowerMode_ )
  {
    LowPowerMode_->UpdateILEInterface( nullptr, nullptr );
  }
  if ( ActiveBlanking_ )
  {
    ActiveBlanking_->UpdateILEInterface( nullptr );
  }
  if ( Ports_ )
  {
    Ports_->UpdateILEInterface( nullptr, nullptr );
  }
}

int CDualILE::ReconnectILEInterfaces()
{
  LogMessage( "Reconnecting to Dual ILE", true );
  IALC_REV_ILE2* vILE2 = ILEWrapper_->GetILEInterface2( ILEDevice_ );
  IALC_REV_ILE4* vILE4 = ILEWrapper_->GetILEInterface4( ILEDevice_ );
  IALC_REVObject3 *vILEDevice1, *vILEDevice2;
  int vRet = DEVICE_OK;
  if ( vILE2 != nullptr )
  {
    if ( vILE2->GetInterface( &vILEDevice1, &vILEDevice2 ) )
    {
      if ( Ports_ != nullptr )
      {
        IALC_REV_Port* vDualPortInterface = ILEDevice_->GetPortInterface();
        if ( vDualPortInterface != nullptr )
        {
          LogMessage( "Reconnecting Dual Ports", true );
          vRet = Ports_->UpdateILEInterface( vDualPortInterface, vILE2 );
        }
        else
        {
          LogMessage( "Pointer to dual port interface invalid" );
          vRet = ERR_DEVICE_RECONNECTIONFAILED;
        }
      }

      if ( vRet == DEVICE_OK && LowPowerMode_ != nullptr )
      {
        IALC_REV_ILEPowerManagement* vUnit1LowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( vILEDevice1 );
        IALC_REV_ILEPowerManagement* vUnit2LowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( vILEDevice2 );
        if ( vUnit1LowPowerMode != nullptr || vUnit2LowPowerMode != nullptr )
        {
          LogMessage( "Reconnecting Dual Low Power Mode", true );
          bool vLowPowerModePresent;
          if ( vUnit1LowPowerMode != nullptr )
          {
            if ( vUnit1LowPowerMode->IsLowPowerPresent( &vLowPowerModePresent ) )
            {
              if ( !vLowPowerModePresent )
              {
                vUnit1LowPowerMode = nullptr;
              }
            }
            else
            {
              vRet = ERR_LOWPOWERPRESENT;
            }
          }
          if ( vRet == DEVICE_OK && vUnit2LowPowerMode != nullptr )
          {
            if ( vUnit2LowPowerMode->IsLowPowerPresent( &vLowPowerModePresent ) )
            {
              if ( !vLowPowerModePresent )
              {
                vUnit2LowPowerMode = nullptr;
              }
            }
            else
            {
              vRet = ERR_LOWPOWERPRESENT;
            }
          }
          if ( vRet == DEVICE_OK )
          {
            vRet = LowPowerMode_->UpdateILEInterface( vUnit1LowPowerMode, vUnit2LowPowerMode );
          }
        }
        else
        {
          LogMessage( "Pointers to both power management interfaces invalid" );
          vRet = ERR_DEVICE_RECONNECTIONFAILED;
        }
      }
    }
    else
    {
      vRet = ERR_DUALILE_GETINTERFACE;
    }
  }
  else
  {
    LogMessage( "Pointer to ILE interface 2 invalid" );
    vRet = ERR_DEVICE_RECONNECTIONFAILED;
  }

  if ( vRet == DEVICE_OK && ActiveBlanking_ != nullptr )
  {
    if ( vILE4 != nullptr )
    {
      LogMessage( "Reconnecting Dual Active Blanking", true );
      vRet = ActiveBlanking_->UpdateILEInterface( vILE4 );
    }
    else
    {
      LogMessage( "Pointer to ILE interface 4 invalid" );
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  return vRet;
}

int CDualILE::InitializePorts()
{
  // Initialisation of Ports configuration
  PortsConfiguration_ = new CPortsConfiguration( DevicesNames_[0], DevicesNames_[1], this );

  IALC_REV_ILE2* vILE2 = ILEWrapper_->GetILEInterface2( ILEDevice_ );
  IALC_REVObject3 *vILEDevice1, *vILEDevice2;
  if ( vILE2->GetInterface( &vILEDevice1, &vILEDevice2 ) )
  {
    IALC_REV_Port* vDualPortInterface = ILEDevice_->GetPortInterface();
    if ( vDualPortInterface!= nullptr )
    {
      try
      {
        Ports_ = new CDualILEPorts( vDualPortInterface, vILE2, PortsConfiguration_, this );
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
  }
  else
  {
    LogMessage( "Retrieving Dual port intefaces failed" );
  }
  return DEVICE_OK;
}

int CDualILE::InitializeActiveBlanking()
{
  IALC_REV_ILE4* vILE4 = ILEWrapper_->GetILEInterface4( ILEDevice_ );
  if ( vILE4 != nullptr )
  {
    bool vUnit1ActiveBlankingPresent = false;
    if ( !vILE4->IsActiveBlankingManagementPresent( 0, &vUnit1ActiveBlankingPresent ) )
    {
      LogMessage( "Dual Active Blanking IsActiveBlankingManagementPresent failed for unit1" );
      return ERR_ACTIVEBLANKING_INIT;
    }
    bool vUnit2ActiveBlankingPresent = false;
    if ( !vILE4->IsActiveBlankingManagementPresent( 1, &vUnit2ActiveBlankingPresent ) )
    {
      LogMessage( "Dual Active Blanking IsActiveBlankingManagementPresent failed for unit2" );
      return ERR_ACTIVEBLANKING_INIT;
    }
    if ( vUnit1ActiveBlankingPresent || vUnit2ActiveBlankingPresent )
    {
      try
      {
        ActiveBlanking_ = new CDualILEActiveBlanking( vILE4, PortsConfiguration_, this );
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
    LogMessage( "Dual Active Blanking interface pointer invalid" );
  }
  return DEVICE_OK;
}

int CDualILE::InitializeLowPowerMode()
{
  IALC_REV_ILE2* vILE2 = ILEWrapper_->GetILEInterface2( ILEDevice_ );
  IALC_REVObject3 *vILEDevice1, *vILEDevice2;
  if ( vILE2->GetInterface( &vILEDevice1, &vILEDevice2 ) )
  {
    IALC_REV_ILEPowerManagement* vUnit1LowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( vILEDevice1 );
    IALC_REV_ILEPowerManagement* vUnit2LowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( vILEDevice2 );
    if ( vUnit1LowPowerMode != nullptr && vUnit2LowPowerMode != nullptr )
    {
      bool vUnit1LowPowerModePresent = false, vUnit2LowPowerModePresent = false;
      if ( !vUnit1LowPowerMode->IsLowPowerPresent( &vUnit1LowPowerModePresent ) )
      {
        LogMessage( "ILE Power IsLowPowerPresent failed for unit1" );
        return ERR_LOWPOWERMODE_INIT;
      }
      if ( !vUnit2LowPowerMode->IsLowPowerPresent( &vUnit2LowPowerModePresent ) )
      {
        LogMessage( "ILE Power IsLowPowerPresent failed for unit2" );
        return ERR_LOWPOWERMODE_INIT;
      }
      if ( vUnit1LowPowerModePresent || vUnit2LowPowerModePresent )
      {
        if ( !vUnit1LowPowerModePresent ) vUnit1LowPowerMode = nullptr;
        if ( !vUnit2LowPowerModePresent ) vUnit2LowPowerMode = nullptr;
        try
        {
          LowPowerMode_ = new CDualILELowPowerMode( vUnit1LowPowerMode, vUnit2LowPowerMode, PortsConfiguration_, this );
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
      std::string vUnitsMessage;
      if ( vUnit1LowPowerMode == nullptr )
      {
        vUnitsMessage += "unit1";
      }
      if ( vUnit2LowPowerMode == nullptr )
      {
        if ( !vUnitsMessage.empty() )
        {
          vUnitsMessage += " and ";
        }
        vUnitsMessage += "unit2";
      }
      LogMessage( "ILE Power interface pointer invalid for " );
    }
  }
  else
  {
    LogMessage( "Retrieving Dual port intefaces failed" );
  }
  return DEVICE_OK;
}
