#include "DragonflyStatus.h"

CDragonflyStatus::CDragonflyStatus( IStatusInterface* StatusInterface, CDragonfly* MMDragonfly )
  : StatusInterface_( StatusInterface ),
  MMDragonfly_( MMDragonfly )
{

}

CDragonflyStatus::~CDragonflyStatus()
{

}

bool CDragonflyStatus::IsRFIDPresentForWheel( TWheelIndex WheelIndex ) const
{
  bool vStatus = false;
  unsigned int vStatusCode;
  StatusInterface_->GetStatusCode( &vStatusCode );
  if ( WheelIndex == WheelIndex1 )
  {
    vStatus = (vStatusCode & 0x10) != 0;
  }
  if ( WheelIndex == WheelIndex2 )
  {
    vStatus = (vStatusCode & 0x40) != 0;
  }
  return vStatus;
}

bool CDragonflyStatus::IsRFIDReadForWheel( TWheelIndex WheelIndex ) const
{
  bool vStatus = false;
  unsigned int vStatusCode;
  StatusInterface_->GetStatusCode( &vStatusCode );
  if ( WheelIndex == WheelIndex1 )
  {
    vStatus = !(vStatusCode & 0x20);
  }
  if ( WheelIndex == WheelIndex2 )
  {
    vStatus = !(vStatusCode & 0x80);
  }
  return vStatus;
}
