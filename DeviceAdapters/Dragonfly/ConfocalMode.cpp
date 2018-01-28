#include "ConfocalMode.h"
#include "ComponentInterface.h"
#include "Dragonfly.h"

using namespace std;

const char* const g_ConfocalModePropertyName = "Imaging Mode";
const char* const g_ConfocalModeReadError = "Failed to retrieve Confocal mode";
const char* const g_Widefield = "Widefield";
const char* const g_TIRF = "TIRF";
const char* const g_ConfocalBaseName = "Confocal";

CConfocalMode::CConfocalMode( IConfocalModeInterface3* ConfocalModeInterface, CDragonfly* MMDragonfly )
  : ConfocalModeInterface_( ConfocalModeInterface ),
  MMDragonfly_( MMDragonfly )
{
  TConfocalMode vCurrentMode;
  bool vValueRetrieved = ConfocalModeInterface_->GetMode( vCurrentMode );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_ConfocalModeReadError );
  }

  // Create property
  CPropertyAction* vAct = new CPropertyAction( this, &CConfocalMode::OnValueChange );
  int vRet = MMDragonfly_->CreateProperty( g_ConfocalModePropertyName, "Undefined", MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + string( g_ConfocalModePropertyName ) + " property" );
  }


  // Populate the possible positions
  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmWideField ) )
  {
    MMDragonfly_->AddAllowedValue( g_ConfocalModePropertyName, g_Widefield );
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmTIRF ) )
  {
    MMDragonfly_->AddAllowedValue( g_ConfocalModePropertyName, g_TIRF );
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmConfocalHC ) )
  {
    ConfocalHCName_ = string( g_ConfocalBaseName ) + " HC";
    int vPinHoleSize;
    if ( ConfocalModeInterface_->GetPinHoleSize_um( bfmConfocalHC, &vPinHoleSize ) )
    {
      ConfocalHCName_ = string( g_ConfocalBaseName ) + " " + to_string( vPinHoleSize ) + "mm";
    }
    MMDragonfly_->AddAllowedValue( g_ConfocalModePropertyName, ConfocalHCName_.c_str() );
  }

  if ( ConfocalModeInterface_->IsConfocalModeAvailable( bfmConfocalHS ) )
  {
    ConfocalHSName_ = string( g_ConfocalBaseName ) + " HS";
    int vPinHoleSize;
    if ( ConfocalModeInterface_->GetPinHoleSize_um( bfmConfocalHS, &vPinHoleSize ) )
    {
      ConfocalHSName_ = string( g_ConfocalBaseName ) + " " + to_string( vPinHoleSize ) + "mm";
    }
    MMDragonfly_->AddAllowedValue( g_ConfocalModePropertyName, ConfocalHSName_.c_str() );
  }

  // Initialise the mode
  switch ( vCurrentMode )
  {
  case bfmWideField:  MMDragonfly_->SetProperty( g_ConfocalModePropertyName, g_Widefield );             break;
  case bfmTIRF:       MMDragonfly_->SetProperty( g_ConfocalModePropertyName, g_TIRF );                  break;
  case bfmConfocalHC: MMDragonfly_->SetProperty( g_ConfocalModePropertyName, ConfocalHCName_.c_str() ); break;
  case bfmConfocalHS: MMDragonfly_->SetProperty( g_ConfocalModePropertyName, ConfocalHSName_.c_str() ); break;
  default:            MMDragonfly_->SetProperty( g_ConfocalModePropertyName, "Undefined" );             break;
  }

}

CConfocalMode::~CConfocalMode()
{

}

int CConfocalMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
  {
    string vRequestedMode;
    Prop->Get( vRequestedMode );
    if ( vRequestedMode == g_Widefield )
    {
      ConfocalModeInterface_->ModeWideField();
    }
    else if ( vRequestedMode == g_TIRF )
    {
      ConfocalModeInterface_->ModeTIRF();
    }
    else if ( vRequestedMode == ConfocalHCName_ )
    {
      ConfocalModeInterface_->ModeConfocalHC();
    }
    else if ( vRequestedMode == ConfocalHSName_ )
    {
      ConfocalModeInterface_->ModeConfocalHS();
    }
  }

  return DEVICE_OK;
}