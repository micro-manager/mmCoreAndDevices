#include "Disk.h"

#include "ComponentInterface.h"
#include "ConfigFileHandler.h"
#include "Dragonfly.h"
#include "DiskStatus.h"

using namespace std;

const char* const g_DiskSpeedPropertyName = "Disk Speed";
const char* const g_DiskStatusPropertyName = "Disk Start/Stop";
const char* const g_DiskSpeedMonitorPropertyName = "Disk Speed Status";
const char* const g_DiskStatusMonitorPropertyName = "Disk Status";
const char* const g_FrameScanTimePropertyName = "Disk Frame Scan Time (ms)";

const char* const g_DiskSpeedLimitsReadError = "Failed to retrieve Disk speed limits";
const char* const g_DiskSpeedValueReadError = "Failed to retrieve the current Disk speed";
const char* const g_DiskSpeedScanTimeReadError = "Failed to retrieve scans per revolution";
const char* const g_DiskStartError = "Failed starting the disk";
const char* const g_DiskSpeedSetError = "Failed setting the speed of the disk";

const char* const g_DiskStatusStop = "Stop";
const char* const g_DiskStatusStart = "Start";

const char* const g_DiskStatusStopped = "Stopped";
const char* const g_DiskStatusReady = "Ready";
const char* const g_DiskStatusChangingSpeed = "Changing speed";
const char* const g_DiskStatusSpeedNotChanging = "Error: Disk speed not changing";
const char* const g_DiskStatusUndefined = "Undefined";

CDisk::CDisk( IDiskInterface2* DiskSpeedInterface, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly )
  : DiskInterface_( DiskSpeedInterface ),
  ConfigFileHandler_( ConfigFileHandler ),
  MMDragonfly_( MMDragonfly ),
  DiskStatusMonitor_( nullptr ),
  ScansPerRevolution_( 0 ),
  SpeedMonitorStateChangeObserver_( nullptr ),
  StatusMonitorStateChangeObserver_( nullptr ),
  FrameScanTimeStateChangeObserver_( nullptr ),
  DiskSimulator_( DiskInterface_, MMDragonfly_),
  DiskStatus_( new CDiskStatus( DiskInterface_, MMDragonfly_, &DiskSimulator_ ) )
{
  // Retrieve initial values
  unsigned int vMin, vMax;
  bool vValueRetrieved = DiskInterface_->GetLimits( vMin, vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedLimitsReadError );
  }

  vValueRetrieved = DiskInterface_->GetScansPerRevolution( &ScansPerRevolution_ );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedScanTimeReadError );
  }

  unsigned int vRequestedSpeed_ = vMax;
  string vSpeedString;
  bool vModeLoadedFromFile = ConfigFileHandler_->LoadPropertyValue( g_DiskSpeedPropertyName, vSpeedString );
  if ( vModeLoadedFromFile )
  {
    try
    {
      vRequestedSpeed_ = stoi( vSpeedString );
    }
    catch ( ... )
    {
      vRequestedSpeed_ = vMax;
    }
  }

  // Start the disk
  if ( !DiskInterface_->IsSpinning() )
  {
    bool vSuccess = DiskInterface_->Start();
    if ( !vSuccess )
    {
      throw std::runtime_error( g_DiskStartError );
    }
  }
  DiskStatus_->Start();

  // Initialise the device speed
  bool vSuccess = DiskSimulator_.SetSpeed( vRequestedSpeed_ );
  //vSuccess = DiskInterface_->SetSpeed( RequestedSpeed_ );
  if ( !vSuccess )
  {
    throw std::runtime_error( g_DiskSpeedSetError );
  }
  DiskStatus_->ChangeSpeed( vRequestedSpeed_ );

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CDisk::OnSpeedChange );
  int vRet = MMDragonfly_->CreateIntegerProperty( g_DiskSpeedPropertyName, vRequestedSpeed_, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + string( g_DiskSpeedPropertyName ) + " property" );
  }
  MMDragonfly_->SetPropertyLimits( g_DiskSpeedPropertyName, vMin, vMax );

  // Create the MM property for Disk status
  vAct = new CPropertyAction( this, &CDisk::OnStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskStatusPropertyName, g_DiskStatusStart, MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    // Not throwing here since it would delete the class and mess up with the already created property
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
  SpeedMonitorStateChangeObserver_ = new CDiskStateChange();
  DiskStatus_->RegisterObserver( SpeedMonitorStateChangeObserver_ );

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskStatusMonitorPropertyName, g_DiskStatusUndefined, MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_DiskStatusMonitorPropertyName ) + " property" );
    return;
  }
  StatusMonitorStateChangeObserver_ = new CDiskStateChange();
  DiskStatus_->RegisterObserver( StatusMonitorStateChangeObserver_ );

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_FrameScanTimePropertyName, g_DiskStatusUndefined, MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_FrameScanTimePropertyName ) + " property" );
    return;
  }
  FrameScanTimeStateChangeObserver_ = new CDiskStateChange();
  DiskStatus_->RegisterObserver( FrameScanTimeStateChangeObserver_ );

  // Start the disk status monitor thread
  DiskStatusMonitor_ = new CDiskStatusMonitor( MMDragonfly_, DiskStatus_ );
  DiskStatusMonitor_->activate();
}

CDisk::~CDisk()
{
  delete DiskStatusMonitor_;
  delete DiskStatus_;
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
          if ( DiskSimulator_.SetSpeed( vRequestedSpeed ) )
          //if ( DiskInterface_->SetSpeed( vRequestedSpeed ) )
          {
            ConfigFileHandler_->SavePropertyValue( g_DiskSpeedPropertyName, to_string( vRequestedSpeed ) );
            DiskStatus_->ChangeSpeed( vRequestedSpeed );
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
        DiskStatus_->Start();
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
        DiskStatus_->Stop();
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
    
    if ( Prop->GetName() == g_DiskSpeedMonitorPropertyName )
    {
      // Update speed
      bool StateChanged = SpeedMonitorStateChangeObserver_->HasBeenNotified();
      if ( DiskStatus_->IsChangingSpeed() )
      {
        unsigned int vDeviceSpeed = DiskStatus_->GetCurrentSpeed();
        Prop->Set( (long)vDeviceSpeed );
        MMDragonfly_->UpdatePropertyUI( g_DiskSpeedMonitorPropertyName, to_string( vDeviceSpeed ).c_str() );
      }
      else if ( StateChanged && DiskStatus_->IsAtSpeed() )
      {
        unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
        Prop->Set( (long)vRequestedSpeed );
        MMDragonfly_->UpdatePropertyUI( g_DiskSpeedMonitorPropertyName, to_string( vRequestedSpeed ).c_str() );
      }
    }
    else if ( Prop->GetName() == g_DiskStatusMonitorPropertyName )
    {
      // Update status
//NOTE: Handle error case where speed is unchanged and not at requested speed
      if ( StatusMonitorStateChangeObserver_->HasBeenNotified() )
      {
        if ( DiskStatus_->IsChangingSpeed() || DiskStatus_->IsStopping() )
        {
          Prop->Set( g_DiskStatusChangingSpeed );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusChangingSpeed );
        }
        else if ( DiskStatus_->IsAtSpeed() )
        {
          Prop->Set( g_DiskStatusReady );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusReady );
        }
        else if ( DiskStatus_->IsStopped() )
        {
          Prop->Set( g_DiskStatusStopped );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusStopped );
        }
      }
    }
    else if ( Prop->GetName() == g_FrameScanTimePropertyName )
    {
      // Update frame scan time
      if ( FrameScanTimeStateChangeObserver_->HasBeenNotified() )
      {
        if ( DiskStatus_->IsStopping() )
        {
          Prop->Set( g_DiskStatusUndefined );
          MMDragonfly_->UpdatePropertyUI( g_FrameScanTimePropertyName, g_DiskStatusUndefined );
        }
        else if ( DiskStatus_->IsChangingSpeed() )
        {
          double vFrameScanTime = CalculateFrameScanTime( DiskStatus_->GetRequestedSpeed(), ScansPerRevolution_ );
          Prop->Set( vFrameScanTime );
          string vStringValue;
          Prop->Get( vStringValue );
          MMDragonfly_->UpdatePropertyUI( g_FrameScanTimePropertyName, vStringValue.c_str() );
        }
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

CDiskStatusMonitor::CDiskStatusMonitor( CDragonfly* MMDragonfly, CDiskStatus* DiskStatus )
  :MMDragonfly_( MMDragonfly ),
  DiskStatus_( DiskStatus ),
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
    DiskStatus_->UpdateFromDevice();
    MMDragonfly_->UpdateProperty( g_DiskSpeedMonitorPropertyName );
    MMDragonfly_->UpdateProperty( g_DiskStatusMonitorPropertyName );
    MMDragonfly_->UpdateProperty( g_FrameScanTimePropertyName );
    Sleep( 500 );
  }
  return 0;
}
