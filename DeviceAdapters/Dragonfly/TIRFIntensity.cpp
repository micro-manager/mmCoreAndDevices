#include "TIRFIntensity.h"

#include "ComponentInterface.h"
#include "ConfocalMode.h"
#include "Dragonfly.h"

const char* const g_TIRFIntensityPropertyName = "TIRF | Optical Feedback";
const char* const g_TIRFIntensityLimitsReadError = "Failed to retrieve TIRF intensity limits";
const char* const g_TIRFIntensityValueReadError = "Failed to retrieve the current TIRF intensity";

CTIRFIntensity::CTIRFIntensity( ITIRFIntensityInterface* TIRFIntensity, IConfocalMode* ConfocalMode, CDragonfly* MMDragonfly )
  : TIRFIntensity_( TIRFIntensity ),
  ConfocalMode_( ConfocalMode ),
  MMDragonfly_( MMDragonfly )
{
  // Retrieve initial values
  if ( !TIRFIntensity_->GetTIRFIntensity( &CurrentTIRFIntensity_ ) )
  {
    throw std::runtime_error( g_TIRFIntensityValueReadError );
  }

  // Create the MM property for TIRF intensity
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFIntensity::OnMonitorStatusChange );
  int vRet = MMDragonfly_->CreateIntegerProperty( g_TIRFIntensityPropertyName, CurrentTIRFIntensity_, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw std::runtime_error( "Error creating " + std::string( g_TIRFIntensityPropertyName ) + " property" );
  }

  // Start the TIRF intensity monitor thread
  TIRFIntensityMonitor_ = std::make_unique<CTIRFIntensityMonitor>( this );
  TIRFIntensityMonitor_->activate();
}

CTIRFIntensity::~CTIRFIntensity()
{
}

void CTIRFIntensity::UpdateFromDevice()
{
  // Only retrieve the TIRF intensity and update the UI if TIRF is selected
  if ( ConfocalMode_->IsTIRFSelected() )
  {
    int vNewPosition;
    if ( TIRFIntensity_->GetTIRFIntensity( &vNewPosition ) )
    {
      int vOldPosition;
      {
        std::lock_guard<std::mutex> lock( TIRFIntensityMutex_ );
        vOldPosition = CurrentTIRFIntensity_;
        CurrentTIRFIntensity_ = vNewPosition;
      }
      // Only refresh the UI if there is any change to report
      if ( vNewPosition != vOldPosition )
      {
        MMDragonfly_->UpdateProperty( g_TIRFIntensityPropertyName );
      }
    }
  }
}

int CTIRFIntensity::OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {    
    if ( Prop->GetName() == g_TIRFIntensityPropertyName )
    {
      // Update TIRF intensity
      std::string vCurrentTIRFIntensityString;
      {
        std::lock_guard<std::mutex> lock( TIRFIntensityMutex_ );
        Prop->Set( static_cast< long >( CurrentTIRFIntensity_ ) );
        vCurrentTIRFIntensityString = std::to_string( CurrentTIRFIntensity_ );
      }
      MMDragonfly_->UpdatePropertyUI( g_TIRFIntensityPropertyName, vCurrentTIRFIntensityString.c_str() );
    }
  }
  return vRet;
}

///////////////////////////////////////////////////////////////////////////////
// TIRF intensity monitoring thread
///////////////////////////////////////////////////////////////////////////////

CTIRFIntensityMonitor::CTIRFIntensityMonitor( CTIRFIntensity* TIRFIntensity )
  :TIRFIntensity_( TIRFIntensity ),
  KeepRunning_( true )
{
}

CTIRFIntensityMonitor::~CTIRFIntensityMonitor()
{
  KeepRunning_ = false;
  wait();
}

int CTIRFIntensityMonitor::svc()
{
  while ( KeepRunning_ )
  {
    TIRFIntensity_->UpdateFromDevice();
    Sleep( 1000 );
  }
  return 0;
}
