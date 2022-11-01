#include "BTIRF.h"
#include "Dragonfly.h"
#include "ASDInterface.h"

#include <stdexcept>


CBTIRF::CBTIRF( IASDInterface4* ASDInterface4, CDragonfly* MMDragonfly )
  : MMDragonfly_( MMDragonfly )
{
  // BTIRF60
  if ( ASDInterface4->IsBorealisTIRF60Available() && ASDInterface4->GetBorealisTIRF60() )
  {
    IBorealisTIRFInterface* vBTIRFInterface = ASDInterface4->GetBorealisTIRF60();
    BTIRF60CriticalAngle_ = std::make_unique<CBTIRFCriticalAngleProperty>( vBTIRFInterface, MMDragonfly_ );
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "BTIRF60 not available" );
  }

  // BTIRF100
  if ( ASDInterface4->IsBorealisTIRF100Available() && ASDInterface4->GetBorealisTIRF100() )
  {
    IBorealisTIRFInterface* vBTIRFInterface = ASDInterface4->GetBorealisTIRF100();
    BTIRF100CriticalAngle_ = std::make_unique<CBTIRFCriticalAngleProperty>( vBTIRFInterface, MMDragonfly_ );
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "BTIRF100 not available" );
  }
}

CBTIRF::~CBTIRF()
{
}

///////////////////////////////////////////////////////////////////////////////
// CBTIRFCriticalAngleProperty
///////////////////////////////////////////////////////////////////////////////

CBTIRFCriticalAngleProperty::CBTIRFCriticalAngleProperty( IBorealisTIRFInterface* BTIRFInterface, CDragonfly* MMDragonfly )
  : MMDragonfly_( MMDragonfly ),
  BTIRFInterface_( BTIRFInterface )
{
  int vMag;
  bool vValueRetrieved = BTIRFInterface_->GetBTMag( &vMag );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve BTIRF Magnification" );
  }

  PropertyName_ = "TIRF | BTIRF" + std::to_string( vMag ) + " Critical Angle";

  int vMin, vMax, vValue = 0;
  vValueRetrieved = BTIRFInterface_->GetBTAngleLimit( &vMin, &vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve " + PropertyName_ + " limits" );
  }

  vValueRetrieved = BTIRFInterface_->GetBTAngle( &vValue );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve the current value for " + PropertyName_ );
  }

  // Create the MM property for Critical angle
  CPropertyAction* vAct = new CPropertyAction( this, &CBTIRFCriticalAngleProperty::OnChange );
  MMDragonfly_->CreateIntegerProperty( PropertyName_.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName_.c_str(), vMin, vMax );
}

CBTIRFCriticalAngleProperty::~CBTIRFCriticalAngleProperty()
{

}

int CBTIRFCriticalAngleProperty::OnChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;

  if ( Act == MM::BeforeGet )
  {
    int vMin, vMax;
    bool vLimitsRetrieved = BTIRFInterface_->GetBTAngleLimit( &vMin, &vMax );
    if ( vLimitsRetrieved )
    {
      Prop->SetLimits( vMin, vMax );
      int vValue;
      if ( BTIRFInterface_->GetBTAngle( &vValue ) )
      {
        Prop->Set( static_cast< long >( vValue ) );
        vRet = DEVICE_OK;
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Failed to retrieve the current value for " + PropertyName_ );
        vRet = DEVICE_ERR;
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to retrieve " + PropertyName_ + " limits" );
      vRet = DEVICE_ERR;
    }
  }
  else if ( Act == MM::AfterSet )
  {
    long vRequestedValue;
    Prop->Get( vRequestedValue );
    MMDragonfly_->LogComponentMessage( "Set BT Angle to [" + std::to_string( vRequestedValue ) + "]", true );
    BTIRFInterface_->SetBTAngle( vRequestedValue );
  }

  return vRet;
}