#include "Disk.h"

#include "ComponentInterface.h"
#include "ConfigFileHandler.h"
#include "Dragonfly.h"

using namespace std;

const char* const g_DiskSpeedPropertyName = "Disk Speed";
const char* const g_DiskStatusPropertyName = "Disk Start/Stop";
const char* const g_DiskSpeedMonitorPropertyName = "Disk Speed Status";
const char* const g_DiskStatusMonitorPropertyName = "Disk Status";
const char* const g_FrameScanTime = "Disk Frame Scan Time (ms)";

const char* const g_DiskSpeedLimitsReadError = "Failed to retrieve Disk speed limits";
const char* const g_DiskSpeedValueReadError = "Failed to retrieve the current Disk speed";
const char* const g_DiskStartError = "Failed starting the disk";
const char* const g_DiskSpeedSetError = "Failed setting the speed of the disk";

const char* const g_DiskStatusStop = "Stop";
const char* const g_DiskStatusStart = "Start";

const char* const g_DiskStatusStopped = "Stopped";
const char* const g_DiskStatusReady = "Ready";
const char* const g_DiskStatusChangingSpeed = "Changing speed";
const char* const g_DiskStatusSpeedNotChanging = "Error: Disk speed not changing";
const char* const g_FrameScanTimeDiskChangingSpeed = "Please wait, disk changing speed";
const char* const g_DiskStatusUndefined = "Undefined";

CDisk::CDisk( IDiskInterface2* DiskSpeedInterface, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly )
  : DiskInterface_( DiskSpeedInterface ),
  ConfigFileHandler_( ConfigFileHandler ),
  MMDragonfly_( MMDragonfly ),
  RequestedSpeed_( 0 ),
  TargetRangeMin_( 0 ),
  TargetRangeMax_ ( 0 ),
  DiskStatusMonitor_( nullptr ),
  RequestedSpeedAchieved_( false ),
  StopRequested_( false ),
  StopWitnessed_( false ),
  FrameScanTimeUpdated_( false ),
  DiskSpeedIncreasing_( true ),
  DiskSpeedStableOnce_( false ),
  DiskSpeedStableTwice_( false ),
  MaxSpeedReached_( 0 ),
  MinSpeedReached_( 0 ),
  DiskSimulator_( DiskInterface_, MMDragonfly_)
{
  // Retrieve initial values
  unsigned int vMin, vMax;
  bool vValueRetrieved = DiskInterface_->GetLimits( vMin, vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( g_DiskSpeedLimitsReadError );
  }

  RequestedSpeed_ = vMax;
  string vSpeedString;
  bool vModeLoadedFromFile = ConfigFileHandler_->LoadPropertyValue( g_DiskSpeedPropertyName, vSpeedString );
  if ( vModeLoadedFromFile )
  {
    try
    {
      RequestedSpeed_ = stoi( vSpeedString );
    }
    catch ( ... )
    {
      RequestedSpeed_ = vMax;
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
  bool vSuccess = DiskSimulator_.GetSpeed( PreviousSpeed_ );
  //bool vSuccess = DiskInterface_->GetSpeed( PreviousSpeed_ );
  if ( !vSuccess )
  {
    throw std::runtime_error( g_DiskSpeedValueReadError );
  }
  vSuccess = DiskSimulator_.SetSpeed( RequestedSpeed_ );
  //vSuccess = DiskInterface_->SetSpeed( RequestedSpeed_ );
  if ( !vSuccess )
  {
    throw std::runtime_error( g_DiskSpeedSetError );
  }
  UpdateSpeedRange();
  DiskSpeedIncreasing_ = PreviousSpeed_ < RequestedSpeed_;
  MaxSpeedReached_ = vMax + 1000;

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CDisk::OnSpeedChange );
  int vRet = MMDragonfly_->CreateIntegerProperty( g_DiskSpeedPropertyName, RequestedSpeed_, false, vAct );
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

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_DiskStatusMonitorPropertyName, g_DiskStatusUndefined, MM::String, true, vAct );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_DiskStatusMonitorPropertyName ) + " property" );
    return;
  }

  vAct = new CPropertyAction( this, &CDisk::OnMonitorStatusChange );
  vRet = MMDragonfly_->CreateProperty( g_FrameScanTime, g_DiskStatusUndefined, MM::String, true, vAct );
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
          if ( DiskSimulator_.SetSpeed( vRequestedSpeed ) )
          //if ( DiskInterface_->SetSpeed( vRequestedSpeed ) )
          {
            RequestedSpeed_ = vRequestedSpeed;
            ConfigFileHandler_->SavePropertyValue( g_DiskSpeedPropertyName, to_string( RequestedSpeed_ ) );
            UpdateSpeedRange();
            if ( !StopRequested_ )
            {
              RequestedSpeedAchieved_ = false;
              FrameScanTimeUpdated_ = false;
              MinSpeedReached_ = 0;
              MaxSpeedReached_ = RequestedSpeed_ + 1000;
              DiskSpeedStableOnce_ = false;
              DiskSpeedStableTwice_ = false;
            }
            if ( DiskSimulator_.GetSpeed( PreviousSpeed_ ) )
            //if ( DiskInterface_->GetSpeed( PreviousSpeed_ ) )
            {
              DiskSpeedIncreasing_ = PreviousSpeed_ < RequestedSpeed_;
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
        StopWitnessed_ = false;
        RequestedSpeedAchieved_ = false;
        FrameScanTimeUpdated_ = false;
        MinSpeedReached_ = 0;
        MaxSpeedReached_ = RequestedSpeed_ + 1000;
        DiskSpeedStableOnce_ = false;
        DiskSpeedStableTwice_ = false;
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

#define _ABSOLUTE_SPEED_RANGE_
void CDisk::UpdateSpeedRange()
{
#ifdef _ABSOLUTE_SPEED_RANGE_
  TargetRangeMin_ = RequestedSpeed_ - 20;
  TargetRangeMax_ = RequestedSpeed_ + 20;
#else
  static unsigned int vDynamicRangePercent = 1;
  TargetRangeMin_ = RequestedSpeed_ * ( 100 - vDynamicRangePercent ) / 100;
  TargetRangeMax_ = RequestedSpeed_ * ( 100 + vDynamicRangePercent ) / 100;
#endif
}

//bool GetSimulatedSpeed( unsigned int& vSimulatedDiskSpeed )
//{
//  bool vValueRetrieved = DiskInterface_->GetSpeed( vSimulatedDiskSpeed );
//  if ( vValueRetrieved )
//  {
//    int vSimulatedError = rand() % 10;
//    int vSign = ( rand() % 2 ) * 2 - 1;
//    vSimulatedDiskSpeed = vSimulatedDiskSpeed * ( 1.f + vSimulatedError * vSign / 100.f );
//  }
//  return vValueRetrieved;
//}

string BoolToString( bool Value )
{
  return ( Value ? "TRUE" : "FALSE" );
}

bool CDisk::IsSpeedWithinMargin( unsigned int CurrentSpeed )
{
  if ( CurrentSpeed > PreviousSpeed_ )
  {
    // Speed is increasing
    if ( !DiskSpeedIncreasing_ )
    {
      // The speed was previously decreasing, we may have reached a local min
      if ( CurrentSpeed > RequestedSpeed_ )
      {
        // Unlikely case where we are above the requested speed
        if ( PreviousSpeed_ >= RequestedSpeed_ )
        {
          // Boolean was badly initialised previously or a new requested value has been set (?)
        }
        else
        {
          // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
          MinSpeedReached_ = PreviousSpeed_;
        }
      }
      else
      {
        // We are below the requested speed therefore we reached a local minimum
        MinSpeedReached_ = min( PreviousSpeed_, CurrentSpeed );
      }
    }
    else
    {
      // Speed continues to increase, we do nothing
    }
    DiskSpeedIncreasing_ = true;
  }
  else if ( CurrentSpeed < PreviousSpeed_ )
  {
    // Speed is decreasing
    if ( DiskSpeedIncreasing_ )
    {
      // The speed was previously increasing, we may have reached a local max
      if ( CurrentSpeed < RequestedSpeed_ )
      {
        // Unlikely case where we are below the requested speed
        if ( PreviousSpeed_ <= RequestedSpeed_ )
        {
          // Boolean was badly initialised previously
        }
        else
        {
          // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
          MaxSpeedReached_ = PreviousSpeed_;
        }
      }
      else
      {
        // We are above the requested speed therefore we reached a local maximum
        MaxSpeedReached_ = max( PreviousSpeed_, CurrentSpeed );
      }
    }
    else
    {
      // Speed continues to decrease, we do nothing
    }
    DiskSpeedIncreasing_ = false;
  }
  else if ( CurrentSpeed == PreviousSpeed_ )
  {
    // Speed hasn't changed
    if ( DiskSpeedStableOnce_ )
    {
      // The speed hasn't changed in 2 ticks
      if ( CurrentSpeed == RequestedSpeed_ )
      {
        // We reached the requested speed
        MinSpeedReached_ = MaxSpeedReached_ = RequestedSpeed_;
      }
      else
      {
        // The disk is not changing speed even though it should
        // Something's wrong, we report it to the user
        DiskSpeedStableTwice_ = true;
      }
    }
    DiskSpeedStableOnce_ = true;
  }
  PreviousSpeed_ = CurrentSpeed;
  if ( MinSpeedReached_ >= TargetRangeMin_ && MaxSpeedReached_ <= TargetRangeMax_ )
  {
    return true;
  }
  return false;
}

int CDisk::OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    // Update speed
    MMDragonfly_->LogComponentMessage( "Status monitor - RequestedSpeedAchieved_ [" + BoolToString( RequestedSpeedAchieved_ ) + "] - StopRequested_ [" + BoolToString( StopRequested_) + "] - StopWitnessed_ [" + BoolToString( StopWitnessed_ ) + "] - FrameScanTimeUpdated_ [" + BoolToString( FrameScanTimeUpdated_ ) + "]" );
    if ( !RequestedSpeedAchieved_ && Prop->GetName() == g_DiskSpeedMonitorPropertyName )
    {
      unsigned int vDeviceSpeed;
      //bool vValueRetrieved = DiskInterface_->GetSpeed( vDeviceSpeed );
      //bool vValueRetrieved = GetSimulatedSpeed( vDeviceSpeed );
      bool vValueRetrieved = DiskSimulator_.GetSpeed( vDeviceSpeed );
      MMDragonfly_->LogComponentMessage( "Disk speed monitor: Speed [" + to_string( vDeviceSpeed ) + "] - value retrieved [" + BoolToString( vValueRetrieved ) + "] - Request [" + to_string( RequestedSpeed_ ) + "]" );
      if ( vValueRetrieved )
      {
        Prop->Set( (long)vDeviceSpeed );
        MMDragonfly_->UpdatePropertyUI( g_DiskSpeedMonitorPropertyName, to_string( vDeviceSpeed ).c_str() );
        if ( ( StopRequested_ && StopWitnessed_ ) || ( !StopRequested_ && IsSpeedWithinMargin( vDeviceSpeed ) ) )
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
      MMDragonfly_->LogComponentMessage( "Disk status monitor" );
      if ( !RequestedSpeedAchieved_ )
      {
        if ( DiskSpeedStableTwice_ )
        {
          Prop->Set( g_DiskStatusSpeedNotChanging );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusSpeedNotChanging );
        }
        else
        {
          Prop->Set( g_DiskStatusChangingSpeed );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusChangingSpeed );
        }
      }
      if ( ( StopRequested_ && !StopWitnessed_ ) )
      {
        if ( !DiskInterface_->IsSpinning() )
        {
          Prop->Set( g_DiskStatusStopped );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusStopped );
          StopWitnessed_ = true;
        }
      }
      if ( RequestedSpeedAchieved_ && !StopRequested_ )
      {
        string vCurrentValue;
        Prop->Get( vCurrentValue );
        if ( vCurrentValue != g_DiskStatusReady )
        {
          Prop->Set( g_DiskStatusReady );
          MMDragonfly_->UpdatePropertyUI( g_DiskStatusMonitorPropertyName, g_DiskStatusReady );
        }
      }
    }
    // Update frame scan time
    if ( Prop->GetName() == g_FrameScanTime )
    {
      MMDragonfly_->LogComponentMessage( "Disk frame scan time monitor" );
      if ( StopRequested_ )
      {
        Prop->Set( g_DiskStatusUndefined );
        MMDragonfly_->UpdatePropertyUI( g_FrameScanTime, g_DiskStatusUndefined );
        FrameScanTimeUpdated_ = true;
      }
      else if ( !RequestedSpeedAchieved_ )
      {
        Prop->Set( g_FrameScanTimeDiskChangingSpeed );
        MMDragonfly_->UpdatePropertyUI( g_FrameScanTime, g_FrameScanTimeDiskChangingSpeed );
      }
      else
      {
        if ( !FrameScanTimeUpdated_ )
        {
          unsigned int vDeviceSpeed;
          //bool vValueRetrieved = DiskInterface_->GetSpeed( vDeviceSpeed );
          bool vValueRetrieved = DiskSimulator_.GetSpeed( vDeviceSpeed );
          if ( vValueRetrieved )
          {
            unsigned int vScansPerResolution;
            vValueRetrieved = DiskInterface_->GetScansPerRevolution( &vScansPerResolution );
            if ( vValueRetrieved )
            {
              double vFrameScanTime = CalculateFrameScanTime( vDeviceSpeed, vScansPerResolution );
              Prop->Set( vFrameScanTime );
              string vStringValue;
              Prop->Get( vStringValue );
              MMDragonfly_->UpdatePropertyUI( g_FrameScanTime, vStringValue.c_str() );
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
