#include "TIRFIntensity.h"

#include "ComponentInterface.h"
#include "Dragonfly.h"

const char* const g_TIRFIntensityPropertyName = "TIRF | Optical Feedback";
const char* const g_TIRFIntensityLimitsReadError = "Failed to retrieve TIRF intensity limits";
const char* const g_TIRFIntensityValueReadError = "Failed to retrieve the current TIRF intensity";

CTIRFIntensity::CTIRFIntensity( ITIRFIntensityInterface* TIRFIntensity, CDragonfly* MMDragonfly )
  : TIRFIntensity_( TIRFIntensity ),
  MMDragonfly_( MMDragonfly )
{
  // Retrieve initial values
  if ( !TIRFIntensity_->GetTIRFIntensityLimit(&TIRFIntensityMin_, &TIRFIntensityMax_) )
  {
    throw std::runtime_error( g_TIRFIntensityLimitsReadError );
  }

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
  MMDragonfly_->SetPropertyLimits( g_TIRFIntensityPropertyName, TIRFIntensityMin_, TIRFIntensityMax_ );

  // Start the TIRF intensity monitor thread
  TIRFIntensityMonitor_ = std::make_unique<CTIRFIntensityMonitor>( MMDragonfly_, this );
  TIRFIntensityMonitor_->activate();
}

CTIRFIntensity::~CTIRFIntensity()
{
}

void CTIRFIntensity::UpdateFromDevice()
{
  int vMin, vMax, vCurrent;
  if ( TIRFIntensity_->GetTIRFIntensityLimit( &vMin, &vMax )
    && TIRFIntensity_->GetTIRFIntensity( &vCurrent ) )
  {
    std::lock_guard<std::mutex> lock( TIRFIntensityMutex_ );
    CurrentTIRFIntensity_ = vCurrent;
    TIRFIntensityMin_ = vMin;
    TIRFIntensityMax_ = vMax;
  }
}

int CTIRFIntensity::OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {    
    if ( Prop->GetName() == g_TIRFIntensityPropertyName )
    {
      // Update Optical Feedback
      std::string vCurrentTIRFIntensityString;
      {
        std::lock_guard<std::mutex> lock( TIRFIntensityMutex_ );
        Prop->SetLimits( TIRFIntensityMin_, TIRFIntensityMax_ );
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

CTIRFIntensityMonitor::CTIRFIntensityMonitor( CDragonfly* MMDragonfly, CTIRFIntensity* TIRFIntensity )
  :MMDragonfly_( MMDragonfly ),
  TIRFIntensity_( TIRFIntensity ),
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
    MMDragonfly_->UpdateProperty( g_TIRFIntensityPropertyName );
    Sleep( 1000 );
  }
  return 0;
}
