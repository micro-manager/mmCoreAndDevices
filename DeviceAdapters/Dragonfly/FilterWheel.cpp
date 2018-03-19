#include "FilterWheel.h"
#include "Dragonfly.h"
#include "FilterWheelProperty.h"
#include "DragonflyStatus.h"
#include "ConfigFileHandlerInterface.h"

#include <stdexcept>

using namespace std;

const char* const g_UnknownMode = "Unknown";
const char* const g_HighSpeedMode = "High Speed";
const char* const g_HighQualityMode = "High Quality";
const char* const g_LowVibrationMode = "Low Vibration";

CFilterWheel::CFilterWheel( TWheelIndex WheelIndex, IFilterWheelInterface* FilterWheelInterface, const CDragonflyStatus* DragonflyStatus, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly )
  : WheelIndex_( WheelIndex ),
  FilterWheelInterface_( FilterWheelInterface ),
  FilterWheelMode_( nullptr ),
  DragonflyStatus_( DragonflyStatus ),
  ConfigFileHandler_( ConfigFileHandler ),
  MMDragonfly_( MMDragonfly ),
  ComponentName_( "Filter Wheel " + to_string( static_cast< long long >( WheelIndex ) ) ),
  FilterModeProperty_( ComponentName_ + " Mode" ),
  RFIDStatusProperty_( ComponentName_ + " RFID Status")
{
  // Initialise critical values
  FilterWheelMode_ = FilterWheelInterface_->GetFilterWheelModeInterface();
  if ( FilterWheelMode_ == nullptr )
  {
    throw runtime_error( "Failed to retrieve the mode's interface for " + ComponentName_ );
    return;
  }

  // Create and initialise the filter wheel property
  FilterWheelProperty_ = new CFilterWheelProperty( this, MMDragonfly_, ComponentName_ + " Position", ComponentName_ );

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
  string vMode;
  bool vModeLoadedFromFile = ConfigFileHandler_->LoadPropertyValue( FilterModeProperty_, vMode );
  if ( vModeLoadedFromFile )
  {
    // Test the validity of the loaded value
    int vModeInteger = GetModeFromString( vMode );
    if ( vModeInteger == FWMUnknown )
    {
      vModeLoadedFromFile = false;
      MMDragonfly_->LogComponentMessage( "Value loaded from config ini file for " + FilterModeProperty_ + " is invalid. Initialising value with default." );
      MMDragonfly_->LogComponentMessage( "Value loaded [" + vMode + "]" ); // Logging the loaded value in a separate call in case the string is garbage
    }
  }
  if ( !vModeLoadedFromFile )
  {
    vMode = GetStringFromMode( FWMHighQuality );
    ConfigFileHandler_->SavePropertyValue( FilterModeProperty_, vMode );
  }

  // Create the MM mode property
  CPropertyAction* vAct = new CPropertyAction( this, &CFilterWheel::OnModeChange );
  int vRet = MMDragonfly_->CreateProperty( FilterModeProperty_.c_str(), vMode.c_str(), MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + FilterModeProperty_  + " property" );
    return;
  }
  // Populate the possible modes
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMHighSpeed ) );
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMHighQuality ) );
  MMDragonfly_->AddAllowedValue( FilterModeProperty_.c_str(), GetStringFromMode( FWMLowVibration ) );
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
    int vRet = MMDragonfly_->CreateProperty( RFIDStatusProperty_.c_str(), vPropertyValue, MM::String, true );
    if ( vRet != DEVICE_OK )
    {
      MMDragonfly_->LogComponentMessage( "Error creating " + RFIDStatusProperty_ + " property" );
    }
  }
  else
  {
    throw logic_error( "Dragonfly status not initialised before " + ComponentName_ );
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
string CFilterWheel::ParseDescription( const string& Description )
{
  // input format:  Name;WavelengthBandwidth1[/WavelengthBandwidth2][/WavelengthBandwidth3] with WavelengthBandwidth in format wavelength-bandwidth
  // output format: wavelength1/wavelength2/wavelength3
  string vOutputDescription;  
  size_t vWavelengthPos = Description.find_first_of( ";" );
  while ( vWavelengthPos != string::npos )
  {
    size_t vBandwidthPos = Description.find_first_of( "-", vWavelengthPos + 1 );
    size_t vNextWavelengthPos = Description.find_first_of( "/", vWavelengthPos + 1 );
    size_t vEndOfWavelength = vBandwidthPos - 1;
    if ( vEndOfWavelength == string::npos )
    {
      vEndOfWavelength = vNextWavelengthPos - 1;
      if ( vEndOfWavelength == string::npos )
      {
        vEndOfWavelength = Description.size();
      }
    }
    if ( !vOutputDescription.empty() )
    {
      vOutputDescription += "/";
    }
    vOutputDescription += Description.substr( vWavelengthPos + 1, vEndOfWavelength - vWavelengthPos );
    vWavelengthPos = vNextWavelengthPos;
  }
  if ( vOutputDescription.empty() )
  {
    vOutputDescription = Description;
  }
  return vOutputDescription;
}

int CFilterWheel::OnModeChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    string vNewMode;
    Prop->Get( vNewMode );
    if ( FilterWheelMode_->SetMode( GetModeFromString( vNewMode ) ) )
    {
      ConfigFileHandler_->SavePropertyValue( FilterModeProperty_, vNewMode );
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to set Filter wheel mode for wheel " + to_string( static_cast< long long >( WheelIndex_ ) ) + " [" + vNewMode + "]" );
      vRet = DEVICE_CAN_NOT_SET_PROPERTY;
    }
  }
  return vRet;
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
    MMDragonfly_->LogComponentMessage( "Undefined filter wheel mode [" + to_string( static_cast< long long >( Mode ) ) + "] retrieved from device for " + ComponentName_ );
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