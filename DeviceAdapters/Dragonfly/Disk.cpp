#include "Disk.h"

#include "ComponentInterface.h"
#include "Dragonfly.h"

using namespace std;

const char* const g_DiskSpeedPropertyName = "Disk Speed";
const char* const g_DiskStatusPropertyName = "Disk Start/Stop";
const char* const g_DiskSpeedMonitorPropertyName = "Disk Speed Status";
const char* const g_DiskStatusMonitorPropertyName = "Disk Status";

const char* const g_DiskSpeedLimitsReadError = "Failed to retrieve Disk speed limits";
const char* const g_DiskSpeedValueReadError = "Failed to retrieve the current Disk speed";

const char* const g_DiskStatusStop = "Stop";
const char* const g_DiskStatusStart = "Start";

const char* const g_DiskStatusStopped = "Stopped";
const char* const g_DiskStatusReady = "Ready";
const char* const g_DiskStatusChangingSpeed = "Changing speed";
const char* const g_DiskStatusUndefined = "Undefined";

CDisk::CDisk( IDiskInterface2* DiskSpeedInterface, CDragonfly* MMDragonfly )
  : DiskInterface_( DiskSpeedInterface ),
  MMDragonfly_( MMDragonfly ),
  CurrentRequestedSpeed_( 0 )
{
  unsigned int vMin, vMax, vSpeed;
  bool vValueRetrieved = DiskInterface_->GetLimits( vMin, vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedLimitsReadError );
  }
  vValueRetrieved = DiskInterface_->GetSpeed( vSpeed );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedValueReadError );
  }

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CDisk::OnSpeedChange );
  MMDragonfly_->CreateIntegerProperty( g_DiskSpeedPropertyName, vSpeed, false, vAct );
  MMDragonfly_->SetPropertyLimits( g_DiskSpeedPropertyName, vMin, vMax );

  // Create the MM property for Disk status
  vAct = new CPropertyAction( this, &CDisk::OnStatusChange );
  MMDragonfly_->CreateProperty( g_DiskStatusPropertyName, g_DiskStatusStop, MM::String, false, vAct );
  MMDragonfly_->AddAllowedValue( g_DiskStatusPropertyName, g_DiskStatusStop );
  MMDragonfly_->AddAllowedValue( g_DiskStatusPropertyName, g_DiskStatusStart );
  // if the disk is spinning, display start as the currently selected option
  if ( DiskInterface_->IsSpinning() )
  {
    MMDragonfly_->SetProperty( g_DiskStatusPropertyName, g_DiskStatusStart );
  }

  // Create the MM properties for Disk status monitor
  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  MMDragonfly_->CreateProperty( g_DiskSpeedMonitorPropertyName, g_DiskStatusUndefined, MM::String, false, vAct );
  MMDragonfly_->CreateProperty( g_DiskStatusMonitorPropertyName, g_DiskStatusUndefined, MM::String, false, vAct );
  DiskStatusMonitor_ = new CDiskStatusMonitor( MMDragonfly_ );
  DiskStatusMonitor_->activate();
}

CDisk::~CDisk()
{
  delete DiskStatusMonitor_;
}

int CDisk::OnSpeedChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    if ( !SetPropertyValueFromDeviceValue( Prop ) )
    {
      return DEVICE_ERR;
    }
  }
  else if ( Act == MM::AfterSet )
  {
    long vRequestedSpeed;
    Prop->Get( vRequestedSpeed );
    if ( vRequestedSpeed >= 0 )
    {
      unsigned int vMin, vMax;
      bool vLimitsRetrieved = DiskInterface_->GetLimits( vMin, vMax );
      if ( vLimitsRetrieved )
      {
        if ( vRequestedSpeed >= (long)vMin && vRequestedSpeed <= (long)vMax )
        {
          DiskInterface_->SetSpeed( vRequestedSpeed );
        }
        else
        {
          MMDragonfly_->LogComponentMessage( "Requested Disk speed is out of bound. Ignoring request." );
        }
      }
      else
      {
        MMDragonfly_->LogComponentMessage( g_DiskSpeedLimitsReadError );
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Requested Disk speed is negavite. Ignoring request." );
    }
  }

  return DEVICE_OK;
}

bool CDisk::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  unsigned int vMin, vMax;
  bool vLimitsRetrieved = DiskInterface_->GetLimits( vMin, vMax );
  if ( vLimitsRetrieved )
  {
    Prop->SetLimits( vMin, vMax );
    unsigned int vSpeed;
    if ( DiskInterface_->GetSpeed( vSpeed ) )
    {
      Prop->Set( (long)vSpeed );
      CurrentRequestedSpeed_ = vSpeed;
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( g_DiskSpeedValueReadError );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( g_DiskSpeedLimitsReadError );
  }

  return vValueSet;
}


int CDisk::OnStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
  }
  else if ( Act == MM::AfterSet )
  {
    string vRequest;
    Prop->Get( vRequest );
    if ( vRequest == g_DiskStatusStart )
    {
      DiskInterface_->Start();
    }
    if ( vRequest == g_DiskStatusStop )
    {
      DiskInterface_->Stop();
    }
  }

  return DEVICE_OK;
}

int CDisk::OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    unsigned int vSpeed;
    bool vValueRetrieved = DiskInterface_->GetSpeed( vSpeed );
    if ( vValueRetrieved )
    {
      // Update speed
      if ( Prop->GetName() == g_DiskSpeedMonitorPropertyName )
      {
        Prop->Set( (long)vSpeed );
      }
      if ( Prop->GetName() == g_DiskStatusMonitorPropertyName )
      {
        // Update status
        if ( !DiskInterface_->IsSpinning() )
        {
          Prop->Set( g_DiskStatusStopped );
        }
        else if ( vSpeed == CurrentRequestedSpeed_ )
        {
          Prop->Set( g_DiskStatusReady );
        }
        else
        {
          Prop->Set( g_DiskStatusChangingSpeed );
        }
      }
    }
    else
    {
      Prop->Set( g_DiskStatusUndefined );
    }
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Disk status monitoring thread
///////////////////////////////////////////////////////////////////////////////

CDiskStatusMonitor::CDiskStatusMonitor( CDragonfly* MMDragonfly )
  :MMDragonfly_( MMDragonfly ),
  KeepRunning_( true )
{
}

CDiskStatusMonitor::~CDiskStatusMonitor()
{
  KeepRunning_ = false;
  wait();
}

int CDiskStatusMonitor::svc()
{
  while ( KeepRunning_ )
  {
    MMDragonfly_->UpdateProperty( g_DiskSpeedMonitorPropertyName );
    MMDragonfly_->UpdateProperty( g_DiskStatusMonitorPropertyName );
    Sleep( 250 );
  }

  return 0;
}
