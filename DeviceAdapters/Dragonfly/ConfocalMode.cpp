#include "ConfocalMode.h"
#include "Dragonfly.h"

using namespace std;

const char* const g_ConfocalModeComposedPropertyName = "Imaging Mode [Power Density]";
const char* const g_ConfocalModeSimplePropertyName = "Imaging Mode";
const char* const g_ConfocalModeReadError = "Failed to retrieve Imaging mode";
const char* const g_PowerDensityPositionsReadError = "Failed to retrieve Power Density positions";
const char* const g_PowerDensityReadError = "Failed to retrieve Power Density current position";
const char* const g_Widefield = "Widefield";
const char* const g_TIRF = "TIRF";
const char* const g_ConfocalBaseName = "Confocal";

CConfocalMode::CConfocalMode( IConfocalModeInterface3* ConfocalModeInterface, IIllLensInterface* IllLensInterface, CDragonfly* MMDragonfly )
  : ConfocalModeInterface_( ConfocalModeInterface ),
  IllLensInterface_( IllLensInterface ),
  MMDragonfly_( MMDragonfly ),
  ConfocalModePropertyName_( g_ConfocalModeComposedPropertyName )
{
  // Retrieve current confocal mode
  TConfocalMode vCurrentConfocalMode;
  bool vValueRetrieved = ConfocalModeInterface_->GetMode( vCurrentConfocalMode );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_ConfocalModeReadError );
  }

  // Retrieve power density values from the device
  CPositionComponentHelper::TPositionNameMap vPowerDensityPositionNames;
  unsigned int vCurrentPowerDensityPosition = 0;
  if ( IllLensInterface_ != nullptr )
  {
    IFilterSet* vFilterSet = IllLensInterface_->GetLensConfigInterface();
    if ( vFilterSet == nullptr )
    {
      MMDragonfly_->LogComponentMessage( "Invalid FilterSet pointer for Power Density" );
    }
    if ( vFilterSet == nullptr || !CPositionComponentHelper::RetrievePositionsFromFilterSet( vFilterSet, vPowerDensityPositionNames, false ) )
    {
      unsigned int vMinValue, vMaxValue;
      if ( IllLensInterface_->GetLimits( vMinValue, vMaxValue ) )
      {
        CPositionComponentHelper::RetrievePositionsWithoutDescriptions( vMinValue, vMaxValue, vPowerDensityPositionNames );
      }
      else
      {
        throw runtime_error( g_PowerDensityPositionsReadError );
      }
    }

    // Retrieve the current power density position from the device
    if ( !IllLensInterface_->GetPosition( vCurrentPowerDensityPosition ) )
    {
      throw runtime_error( g_PowerDensityReadError );
    }
  }
  else
  {
    ConfocalModePropertyName_ = g_ConfocalModeSimplePropertyName;
  }

  // Build the map of possible positions
  string vCurrentModeBaseName;
  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmWideField ) )
  {
    AddValuesForConfocalMode( bfmWideField, g_Widefield, vPowerDensityPositionNames );
    if ( vCurrentConfocalMode == bfmWideField )
    {
      vCurrentModeBaseName = g_Widefield;
    }
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmTIRF ) )
  {
    TDevicePosition vPosition = { bfmTIRF, 0 };
    PositionNameMap_[g_TIRF] = vPosition;
    if ( vCurrentConfocalMode == bfmTIRF )
    {
      vCurrentModeBaseName = g_TIRF;
    }
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmConfocalHC ) )
  {
    ConfocalHCName_ = string( g_ConfocalBaseName ) + " HC";
    int vPinHoleSize;
    if ( ConfocalModeInterface_->GetPinHoleSize_um( bfmConfocalHC, &vPinHoleSize ) )
    {
      ConfocalHCName_ = string( g_ConfocalBaseName ) + " " + to_string( static_cast< long long >( vPinHoleSize ) ) + "mm";
    }
    AddValuesForConfocalMode( bfmConfocalHC, ConfocalHCName_, vPowerDensityPositionNames );
    if ( vCurrentConfocalMode == bfmConfocalHC )
    {
      vCurrentModeBaseName = ConfocalHCName_;
    }
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmConfocalHS ) )
  {
    ConfocalHSName_ = string( g_ConfocalBaseName ) + " HS";
    int vPinHoleSize;
    if ( ConfocalModeInterface_->GetPinHoleSize_um( bfmConfocalHS, &vPinHoleSize ) )
    {
      ConfocalHSName_ = string( g_ConfocalBaseName ) + " " + to_string( static_cast< long long >( vPinHoleSize ) ) + "mm";
    }
    AddValuesForConfocalMode( bfmConfocalHS, ConfocalHSName_, vPowerDensityPositionNames );
    if ( vCurrentConfocalMode == bfmConfocalHS )
    {
      vCurrentModeBaseName = ConfocalHSName_;
    }
  }

  // Reset the confocal mode and power density to their initial value since we modified them
  if ( SetDeviceConfocalMode( vCurrentConfocalMode ) != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Failed to set Imaging mode position [" + to_string( static_cast< long long >( vCurrentConfocalMode ) ) + "]" );
  }
  if ( IllLensInterface_ != nullptr )
  {
    if ( !IllLensInterface_->SetPosition( vCurrentPowerDensityPosition ) )
    {
      MMDragonfly_->LogComponentMessage( "Failed to set Power density position [" + to_string( static_cast< long long >( vCurrentPowerDensityPosition ) ) + "]" );
    }
  }

  // Initialise the property value
  string vCurrentPropertyValue = vCurrentModeBaseName;
  if ( IllLensInterface_ != nullptr )
  {
    string vCurrentPowerDensityName = "Undefined";
    if ( vPowerDensityPositionNames.find( vCurrentPowerDensityPosition ) != vPowerDensityPositionNames.end() )
    {
      vCurrentPowerDensityName = vPowerDensityPositionNames[vCurrentPowerDensityPosition];
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current Power Density position invalid" );
    }
    if ( vCurrentConfocalMode != bfmTIRF )
    {
      vCurrentPropertyValue = BuildPropertyValueFromDeviceValue( vCurrentModeBaseName, vCurrentPowerDensityName );
    }
  }

  // Create property
  CPropertyAction* vAct = new CPropertyAction( this, &CConfocalMode::OnValueChange );
  int vRet = MMDragonfly_->CreateProperty( ConfocalModePropertyName_.c_str(), vCurrentPropertyValue.c_str(), MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + ConfocalModePropertyName_ + " property" );
  }

  // Populate the values
  TPositionNameMap::const_iterator PositionNameIt = PositionNameMap_.begin();
  while ( PositionNameIt != PositionNameMap_.end() )
  {
    MMDragonfly_->AddAllowedValue( ConfocalModePropertyName_.c_str(), PositionNameIt->first.c_str() );
    PositionNameIt++;
  }
}

CConfocalMode::~CConfocalMode()
{

}

#define _CONFOCALMODE_HARDWARE_EXPLORATION_
void CConfocalMode::AddValuesForConfocalMode( TConfocalMode ConfocalMode, const std::string& ConfocalModeBaseName, const CPositionComponentHelper::TPositionNameMap& PowerDensityPositionNames )
{
  if ( IllLensInterface_ == nullptr )
  {
    TDevicePosition vPosition = { ConfocalMode, 0 };
    PositionNameMap_[ConfocalModeBaseName] = vPosition;
    return;
  }
#ifdef _CONFOCALMODE_HARDWARE_EXPLORATION_
  if ( SetDeviceConfocalMode( ConfocalMode ) == DEVICE_OK )
  {
    if ( IllLensInterface_->IsRestrictionEnabled() )
#else
  if ( ConfocalMode == bfmConfocalHC || ConfocalMode == bfmConfocalHS )
#endif
    {
#ifdef _CONFOCALMODE_HARDWARE_EXPLORATION_
      unsigned int vMinPosition;
      unsigned int vMaxPosition;
      if ( IllLensInterface_->GetRestrictedRange( vMinPosition, vMaxPosition ) )
      {
#else
      unsigned int vMinPosition = 1;
      unsigned int vMaxPosition = 2;
#endif
        for ( unsigned int vPositionIndex = vMinPosition; vPositionIndex <= vMaxPosition; ++vPositionIndex )
          {
            CPositionComponentHelper::TPositionNameMap::const_iterator vPosition = PowerDensityPositionNames.find( vPositionIndex );
            if ( vPosition != PowerDensityPositionNames.end() )
            {
              AddValue( ConfocalMode, ConfocalModeBaseName, vPosition->first, vPosition->second );
            }
          }
#ifdef _CONFOCALMODE_HARDWARE_EXPLORATION_
      }
      else
      {
        throw runtime_error( "Failed to read Imaging mode restricted range" );
      }
#endif
    }
    else
    {
      CPositionComponentHelper::TPositionNameMap::const_iterator vPositionIt = PowerDensityPositionNames.begin();
      while ( vPositionIt != PowerDensityPositionNames.end() )
      {
        AddValue( ConfocalMode, ConfocalModeBaseName, vPositionIt->first, vPositionIt->second );
        vPositionIt++;
      }
    }
#ifdef _CONFOCALMODE_HARDWARE_EXPLORATION_
  }  
  else
  {
    throw runtime_error( "Failed to set Imaging mode position [" + to_string( static_cast< long long >( ConfocalMode ) ) + "]" );
  }
#endif
}

string CConfocalMode::BuildPropertyValueFromDeviceValue( const string& ConfocalModeBaseName, const string& PowerDensityName )
{
  return ConfocalModeBaseName + " [" + PowerDensityName + "]";
}

void CConfocalMode::AddValue( TConfocalMode ConfocalMode, const string& ConfocalModeBaseName, unsigned int PowerDensity, const string& PowerDensityName )
{
  TDevicePosition vPosition = { ConfocalMode, PowerDensity };
  string vPropertyValue = BuildPropertyValueFromDeviceValue( ConfocalModeBaseName, PowerDensityName );
  PositionNameMap_[vPropertyValue] = vPosition;
}

int CConfocalMode::SetDeviceConfocalMode( TConfocalMode ConfocalMode )
{
  int vRet = DEVICE_OK;
  bool vDeviceSuccess = true;
  switch ( ConfocalMode )
  {
  case bfmWideField:  vDeviceSuccess = ConfocalModeInterface_->ModeWideField();  break;
  case bfmTIRF:       vDeviceSuccess = ConfocalModeInterface_->ModeTIRF();       break;
  case bfmConfocalHC: vDeviceSuccess = ConfocalModeInterface_->ModeConfocalHC(); break;
  case bfmConfocalHS: vDeviceSuccess = ConfocalModeInterface_->ModeConfocalHS(); break;
  default:            
    MMDragonfly_->LogComponentMessage( "Invalid Imaging mode [" + to_string( static_cast< long long >( ConfocalMode ) ) + "]" );
    vRet = DEVICE_INVALID_PROPERTY_VALUE; 
    break;
  }
  if ( !vDeviceSuccess )
  {
    MMDragonfly_->LogComponentMessage( "Failed to set the Imaging mode" );
    vRet = DEVICE_CAN_NOT_SET_PROPERTY;
  }
  return vRet;
}

int CConfocalMode::SetDeviceFromPropertyValue( const std::string& PropertValue )
{
  int vRet = DEVICE_INVALID_PROPERTY_VALUE;
  if ( PositionNameMap_.find( PropertValue ) != PositionNameMap_.end() )
  {
    TDevicePosition vPosition = PositionNameMap_[PropertValue];
    vRet = SetDeviceConfocalMode( vPosition.ConfocalMode );
    if ( vRet == DEVICE_OK && IllLensInterface_ != nullptr && vPosition.ConfocalMode != bfmTIRF )
    {
      if ( !IllLensInterface_->SetPosition( vPosition.PowerDensity ) )
      {
        MMDragonfly_->LogComponentMessage( "Failed to set the Power density" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
      else
      {
        vRet = DEVICE_OK;
      }
    }
  }
  return vRet;
}

int CConfocalMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    string vRequestedMode;
    Prop->Get( vRequestedMode );
    vRet = SetDeviceFromPropertyValue( vRequestedMode );
  }
  return vRet;
}