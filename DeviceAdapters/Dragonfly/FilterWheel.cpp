#include "FilterWheel.h"
#include "Dragonfly.h"
#include "FilterWheelProperty.h"

#include <stdexcept>

using namespace std;

const char* const g_UnknownMode = "Unknown";
const char* const g_HighSpeedMode = "High Speed";
const char* const g_HighQualityMode = "High Quality";
const char* const g_LowVibrationMode = "Low Vibration";

CFilterWheel::CFilterWheel( TWheelIndex WheelIndex, IFilterWheelInterface* FilterWheelInterface, CDragonfly* MMDragonfly )
  : WheelIndex_( WheelIndex ),
  FilterWheelInterface_( FilterWheelInterface ),
  FilterWheelMode_( nullptr ),
  MMDragonfly_( MMDragonfly ),
  ComponentName_( "Filter Wheel " + to_string( WheelIndex ) ),
  FilterModeProperty_( ComponentName_ + " mode" )
{
  // Retrieve the current mode
  FilterWheelMode_ = FilterWheelInterface_->GetFilterWheelModeInterface();
  if ( FilterWheelMode_ == nullptr )
  {
    throw std::runtime_error( "Failed to retrieve the mode's interface for " + ComponentName_ );
  }
  TFilterWheelMode vMode;
  if ( !FilterWheelMode_->GetMode( vMode ) )
  {
    throw std::runtime_error( "Failed to retrieve the mode of " + ComponentName_ );
  }

  // Create and initialise the filter wheel property
  FilterWheelProperty_ = new CFilterWheelProperty( this, MMDragonfly_, ComponentName_ + " position", ComponentName_ );

  // Create the MM mode property
  CPropertyAction* vAct = new CPropertyAction( this, &CFilterWheel::OnModeChange );
  MMDragonfly_->CreateProperty( FilterModeProperty_.c_str(), "Undefined", MM::String, false, vAct );

  // Populate the possible modes
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMUnknown ) );
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMHighSpeed ) );
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMHighQuality ) );
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMLowVibration ) );

  // Initialise the mode property
  MMDragonfly_->SetProperty( FilterModeProperty_.c_str(), GetStringFromMode( vMode ) );
}

CFilterWheel::~CFilterWheel()
{
  delete FilterWheelProperty_;
}

bool CFilterWheel::GetPosition( unsigned int& Position )
{
  return FilterWheelInterface_->GetPosition( Position );
}
bool CFilterWheel::SetPosition( unsigned int Position )
{
  return FilterWheelInterface_->SetPosition( Position );
}
bool CFilterWheel::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return FilterWheelInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterConfigInterface* CFilterWheel::GetFilterConfigInterface()
{
  return FilterWheelInterface_->GetFilterConfigInterface();
}

int CFilterWheel::OnModeChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    TFilterWheelMode vMode;
    if ( FilterWheelMode_->GetMode( vMode ) )
    {
      Prop->Set( GetStringFromMode( vMode ) );
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to read the current " + FilterModeProperty_ );
    }
  }
  else if ( Act == MM::AfterSet )
  {
    string vNewMode;
    Prop->Get( vNewMode );
    FilterWheelMode_->SetMode( GetModeFromString( vNewMode ) );
  }
  return DEVICE_OK;
}

const char* CFilterWheel::GetStringFromMode( TFilterWheelMode Mode ) const
{
  switch ( Mode )
  {
  case FWMHighSpeed:
    return g_HighSpeedMode;
  case FWMHighQuality:
    return g_HighQualityMode;
  case FWMLowVibration:
    return g_LowVibrationMode;
  case FWMUnknown:
    return g_UnknownMode;
  default:
    MMDragonfly_->LogComponentMessage( "Undefined filter wheel mode [" + to_string( Mode ) + "] retrieved from device for " + ComponentName_ );
    return g_UnknownMode;
  }
}

TFilterWheelMode CFilterWheel::GetModeFromString( const string& ModeString ) const
{
  if ( ModeString == g_HighSpeedMode )
  {
    return FWMHighSpeed;
  }

  if ( ModeString == g_HighQualityMode )
  {
    return FWMHighQuality;
  }

  if ( ModeString == g_LowVibrationMode )
  {
    return FWMLowVibration;
  }

  if ( ModeString == g_UnknownMode )
  {
    return FWMUnknown;
  }

  MMDragonfly_->LogComponentMessage( "Undefined filter wheel mode [" + ModeString + "] requested for " + ComponentName_ );
  return FWMUnknown;
}