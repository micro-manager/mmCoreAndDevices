#include "Disk.h"

#include "ComponentInterface.h"
#include "ConfigFileHandler.h"
#include "Dragonfly.h"
#include "DiskStatusInterface.h"

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
  StatusMonitorStateErrorObserver_( nullptr ),
  DiskSimulator_( DiskInterface_, MMDragonfly_),
  DiskStatus_( CreateDiskStatus( DiskInterface_, MMDragonfly_, &DiskSimulator_ ) )
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
  StatusMonitorStateErrorObserver_ = new CDiskStateError();
  DiskStatus_->RegisterErrorObserver( StatusMonitorStateErrorObserver_ );

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
  DiskStatusMonitor_ = new CDiskStatusMonitor( MMDragonfly_, DiskStatus_, DiskStatusMutex_ );
  DiskStatusMonitor_->activate();
}

CDisk::~CDisk()
{
  delete DiskStatusMonitor_;
  delete DiskStatus_;
  delete SpeedMonitorStateChangeObserver_;
  delete StatusMonitorStateChangeObserver_;
  delete FrameScanTimeStateChangeObserver_;
  delete StatusMonitorStateErrorObserver_;
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
            ConfigFileHandler_->SavePropertyValue( g_DiskSpeedPropertyName, to_string( static_cast< long long >( vRequestedSpeed ) ) );
            boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
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
        boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
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
        boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
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
      string vNewPropertyValue;
      {
        boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
        bool vStateChanged = SpeedMonitorStateChangeObserver_->HasBeenNotified();
        static unsigned int vTick = 0;
        vTick = ( vTick == 5 ? 0 : vTick + 1 );
        if ( DiskStatus_->IsChangingSpeed() && vTick == 0 )
        {
          unsigned int vDeviceSpeed = DiskStatus_->GetCurrentSpeed();
          Prop->Set( (long)vDeviceSpeed );
          vNewPropertyValue = to_string( static_cast< long long >( vDeviceSpeed ) );
        }
        else if ( vStateChanged && DiskStatus_->IsAtSpeed() )
        {
          vTick = 0;
          unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
          Prop->Set( (long)vRequestedSpeed );
          vNewPropertyValue = to_string( static_cast< long long >( vRequestedSpeed ) );
        }
      }
      if ( !vNewPropertyValue.empty() )
      {
        MMDragonfly_->UpdatePropertyUI( g_DiskSpeedMonitorPropertyName, vNewPropertyValue.c_str() );
      }
    }
    else if ( Prop->GetName() == g_DiskStatusMonitorPropertyName )
    {
      // Update status
      string vErrorMessage;
      if ( StatusMonitorStateErrorObserver_->GetErrorMessage( vErrorMessage ) )
      {
        Prop->Set( vErrorMessage.c_str() );
        MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, vErrorMessage.c_str() );
      }
      else if ( StatusMonitorStateChangeObserver_->HasBeenNotified() )
      {
        string vNewPropertyValue;
        {
          boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
          if ( DiskStatus_->IsChangingSpeed() || DiskStatus_->IsStopping() )
          {
            Prop->Set( g_DiskStatusChangingSpeed );
            vNewPropertyValue = g_DiskStatusChangingSpeed;
          }
          else if ( DiskStatus_->IsAtSpeed() )
          {
            Prop->Set( g_DiskStatusReady );
            vNewPropertyValue = g_DiskStatusReady;
          }
          else if ( DiskStatus_->IsStopped() )
          {
            Prop->Set( g_DiskStatusStopped );
            vNewPropertyValue = g_DiskStatusStopped;
          } 
        }
        if ( !vNewPropertyValue.empty() )
        {
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, vNewPropertyValue.c_str() );
        }
      }
    }
    else if ( Prop->GetName() == g_FrameScanTimePropertyName )
    {
      // Update frame scan time
      if ( FrameScanTimeStateChangeObserver_->HasBeenNotified() )
      {
        string vNewPropertyValue;
        {
          boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
          if ( DiskStatus_->IsStopping() )
          {
            Prop->Set( g_DiskStatusUndefined );
            vNewPropertyValue = g_DiskStatusUndefined;
          }
          else if ( DiskStatus_->IsChangingSpeed() )
          {
            double vFrameScanTime = CalculateFrameScanTime( DiskStatus_->GetRequestedSpeed(), ScansPerRevolution_ );
            Prop->Set( vFrameScanTime );
            Prop->Get( vNewPropertyValue );
          }
        }
        if ( !vNewPropertyValue.empty() )
        {
          MMDragonfly_->UpdatePropertyUI( g_FrameScanTimePropertyName, vNewPropertyValue.c_str() );
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

CDiskStatusMonitor::CDiskStatusMonitor( CDragonfly* MMDragonfly, IDiskStatus* DiskStatus, boost::mutex& DiskStatusMutex )
  :MMDragonfly_( MMDragonfly ),
  DiskStatus_( DiskStatus ),
  DiskStatusMutex_( DiskStatusMutex ),
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
    {
      boost::lock_guard<boost::mutex> lock( DiskStatusMutex_ );
      DiskStatus_->UpdateFromDevice();
    }
    MMDragonfly_->UpdateProperty( g_DiskSpeedMonitorPropertyName );
    MMDragonfly_->UpdateProperty( g_DiskStatusMonitorPropertyName );
    MMDragonfly_->UpdateProperty( g_FrameScanTimePropertyName );
    Sleep( 500 );
  }
  return 0;
}
