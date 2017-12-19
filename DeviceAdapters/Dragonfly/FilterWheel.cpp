#include "FilterWheel.h"
#include "Dragonfly.h"
#include "FilterWheelProperty.h"
#include "DragonflyStatus.h"

#include <stdexcept>

using namespace std;

const char* const g_UnknownMode = "Unknown";
const char* const g_HighSpeedMode = "High Speed";
const char* const g_HighQualityMode = "High Quality";
const char* const g_LowVibrationMode = "Low Vibration";

CFilterWheel::CFilterWheel( TWheelIndex WheelIndex, IFilterWheelInterface* FilterWheelInterface, const CDragonflyStatus* DragonflyStatus, CDragonfly* MMDragonfly )
  : WheelIndex_( WheelIndex ),
  FilterWheelInterface_( FilterWheelInterface ),
  FilterWheelMode_( nullptr ),
  DragonflyStatus_( DragonflyStatus ),
  MMDragonfly_( MMDragonfly ),
  ComponentName_( "Filter Wheel " + to_string( WheelIndex ) ),
  FilterModeProperty_( ComponentName_ + " mode" ),
  RFIDStatusProperty_( ComponentName_ + " RFID status")
{
  // Create and initialise the filter wheel property
  FilterWheelProperty_ = new CFilterWheelProperty( this, MMDragonfly_, ComponentName_ + " position", ComponentName_ );

  // Create and initialise the mode property
  CreateModeProperty();
  
  // Create and initialise the RFID status property
  CreateRFIDStatusProperty();
}

CFilterWheel::~CFilterWheel()
{
  delete FilterWheelProperty_;
}

void CFilterWheel::CreateModeProperty()
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

void CFilterWheel::CreateRFIDStatusProperty()
{
  if ( DragonflyStatus_ != nullptr )
  {
    char vPropertyValue[32];
    if ( !DragonflyStatus_->IsRFIDPresentForWheel( WheelIndex_ ) )
    {
      strncpy( vPropertyValue, "Not present", 32 );
    }
    else
    {
      if ( DragonflyStatus_->IsRFIDReadForWheel( WheelIndex_ ) )
      {
        strncpy( vPropertyValue, "Present and Read", 32 );
      }
      else
      {
        strncpy( vPropertyValue, "Present but Read failed", 32 );
      }
    }
    MMDragonfly_->CreateProperty( RFIDStatusProperty_.c_str(), vPropertyValue, MM::String, true );
  }
  else
  {
    throw std::logic_error( "Dragonfly status not initialised before " + ComponentName_ );
  }
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