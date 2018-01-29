#include "Disk.h"

#include "ComponentInterface.h"
#include "Dragonfly.h"

using namespace std;

const char* const g_DiskSpeedPropertyName = "Disk Speed";
const char* const g_DiskStatusPropertyName = "Disk Start/Stop";
const char* const g_DiskSpeedMonitorPropertyName = "Disk Speed Status";
const char* const g_DiskStatusMonitorPropertyName = "Disk Status";
const char* const g_FrameScanTime = "Disk Frame Scan Time (ms)";

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
  RequestedSpeed_( 0 ),
  DiskStatusMonitor_( nullptr ),
  RequestedSpeedAchieved_( false ),
  StopRequested_( false ),
  StopWitnessed_( false ),
  FrameScanTimeUpdated_( false )

{
  unsigned int vMin, vMax, vSpeed, vScansPerRevolution;
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
  RequestedSpeed_ = vSpeed;
  vValueRetrieved = DiskInterface_->GetScansPerRevolution( &vScansPerRevolution );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedValueReadError );
  }

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CDisk::OnSpeedChange );
  int vRet = MMDragonfly_->CreateIntegerProperty( g_DiskSpeedPropertyName, vSpeed, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + string( g_DiskSpeedPropertyName ) + " property" );
  }
  MMDragonfly_->SetPropertyLimits( g_DiskSpeedPropertyName, vMin, vMax );

  // Create the MM property for Disk status
  // if the disk is spinning, display start as the currently selected option
  string vStatusValue = g_DiskStatusStop;
  if ( DiskInterface_->IsSpinning() )
  {
    vStatusValue = g_DiskStatusStart;
  }
  vAct = new CPropertyAction( this, &CDisk::OnStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskStatusPropertyName, vStatusValue.c_str(), MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    // Not throwing here since it would delete the class and mess up with the already create property
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_DiskStatusPropertyName ) + " property" );
    return;
  }
  MMDragonfly_->AddAllowedValue( g_DiskStatusPropertyName, g_DiskStatusStop );
  MMDragonfly_->AddAllowedValue( g_DiskStatusPropertyName, g_DiskStatusStart );


  // Create the MM properties for Disk status monitor
  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskSpeedMonitorPropertyName, g_DiskStatusUndefined, MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_DiskSpeedMonitorPropertyName ) + " property" );
    return;
  }

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskStatusMonitorPropertyName, g_DiskStatusUndefined, MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_DiskStatusMonitorPropertyName ) + " property" );
    return;
  }

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_FrameScanTime, to_string( CalculateFrameScanTime( vSpeed, vScansPerRevolution )).c_str(), MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_FrameScanTime ) + " property" );
    return;
  }

  DiskStatusMonitor_ = new CDiskStatusMonitor( MMDragonfly_ );
  DiskStatusMonitor_->activate();
}

CDisk::~CDisk()
{
  delete DiskStatusMonitor_;
}

int CDisk::OnSpeedChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
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
          if ( DiskInterface_->SetSpeed( vRequestedSpeed ) )
          {
            RequestedSpeed_ = vRequestedSpeed;
            if ( !StopRequested_ )
            {
              RequestedSpeedAchieved_ = false;
              FrameScanTimeUpdated_ = false;
            }
          }
          else
          {
            MMDragonfly_->LogComponentMessage( "Failed to change the speed of the disk" );
            vRet = DEVICE_CAN_NOT_SET_PROPERTY;
          }
        }
        else
        {
          MMDragonfly_->LogComponentMessage( "Requested Disk speed is out of bound. Ignoring request." );
          vRet = DEVICE_INVALID_PROPERTY_VALUE;
        }
      }
      else
      {
        MMDragonfly_->LogComponentMessage( g_DiskSpeedLimitsReadError );
        vRet = DEVICE_ERR;
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Requested Disk speed is negavite. Ignoring request." );
      vRet = DEVICE_INVALID_PROPERTY_VALUE;
    }
  }

  return vRet;
}

int CDisk::OnStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    string vRequest;
    Prop->Get( vRequest );
    if ( vRequest == g_DiskStatusStart )
    {
      if ( DiskInterface_->Start() )
      {
        StopRequested_ = false;
        RequestedSpeedAchieved_ = false;
        FrameScanTimeUpdated_ = false;
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Failed to start the disk" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
    }
    if ( vRequest == g_DiskStatusStop )
    {
      if ( DiskInterface_->Stop() )
      {
        StopRequested_ = true;
        StopWitnessed_ = false;
        RequestedSpeedAchieved_ = false;
        FrameScanTimeUpdated_ = false;
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Failed to stop the disk" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
    }
  }

  return vRet;
}

int CDisk::OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    // Update speed
    if ( !RequestedSpeedAchieved_ && Prop->GetName() == g_DiskSpeedMonitorPropertyName )
    {
      unsigned int vDeviceSpeed;
      bool vValueRetrieved = DiskInterface_->GetSpeed( vDeviceSpeed );
      if ( vValueRetrieved )
      {
        Prop->Set( (long)vDeviceSpeed );
        if ( ( StopRequested_ && StopWitnessed_) || (!StopRequested_ && vDeviceSpeed == RequestedSpeed_) )
        {
          RequestedSpeedAchieved_ = true;
        }
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Failed to read the speed of the disk" );
        vRet = DEVICE_ERR;
      }
    }
    // Update status
    if ( Prop->GetName() == g_DiskStatusMonitorPropertyName )
    {
      if ( !RequestedSpeedAchieved_ )
      {
        Prop->Set( g_DiskStatusChangingSpeed );
      }
      if ( ( StopRequested_ && !StopWitnessed_ ) )
      {
        if ( !DiskInterface_->IsSpinning() )
        {
          Prop->Set( g_DiskStatusStopped );
          StopWitnessed_ = true;
        }
      }
      if ( RequestedSpeedAchieved_ && !StopRequested_ )
      {
        Prop->Set( g_DiskStatusReady );
      }
    }
    // Update frame scan time
    if ( Prop->GetName() == g_FrameScanTime )
    {
      if ( RequestedSpeedAchieved_ && !StopRequested_ )
      {
        if ( !FrameScanTimeUpdated_ )
        {
          unsigned int vDeviceSpeed;
          bool vValueRetrieved = DiskInterface_->GetSpeed( vDeviceSpeed );
          if ( vValueRetrieved )
          {
            unsigned int vScansPerResolution;
            vValueRetrieved = DiskInterface_->GetScansPerRevolution( &vScansPerResolution );
            if ( vValueRetrieved )
            {
              Prop->Set( CalculateFrameScanTime( vDeviceSpeed, vScansPerResolution ) );
              FrameScanTimeUpdated_ = true;
            }
            else
            {
              MMDragonfly_->LogComponentMessage( "Error in call to GetScansPerRevolution()" );
              vRet = DEVICE_ERR;
            }
          }
          else
          {
            MMDragonfly_->LogComponentMessage( "Failed to read the speed of the disk" );
            vRet = DEVICE_ERR;
          }
        }
      }
      else
      {
        Prop->Set( g_DiskStatusUndefined );
      }
    }
  }
  return vRet;
}

double CDisk::CalculateFrameScanTime(unsigned int Speed, unsigned int ScansPerRevolution )
{
  if ( Speed == 0 || ScansPerRevolution == 0 )
  {
    return 0;
  }
  return 60.0 * 1000.0 / (double)( Speed * ScansPerRevolution );
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
    MMDragonfly_->UpdateProperty( g_FrameScanTime );
    Sleep( 500 );
  }
  return 0;
}
