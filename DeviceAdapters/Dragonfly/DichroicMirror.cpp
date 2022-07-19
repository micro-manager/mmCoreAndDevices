#include "DichroicMirror.h"
#include "FilterWheelProperty.h"

#include "ASDInterface.h"

const char* const g_DichroicMirrorPosition = "Dichroic Mirror";

CDichroicMirror::CDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface, CDragonfly* MMDragonfly )
  : DichroicMirrorInterface_( DichroicMirrorInterface ),
  MMDragonfly_( MMDragonfly )
{
  FilterWheelProperty_ = new CFilterWheelProperty( this, MMDragonfly_, g_DichroicMirrorPosition, g_DichroicMirrorPosition );
}

CDichroicMirror::~CDichroicMirror()
{
  delete FilterWheelProperty_;
}

bool CDichroicMirror::GetPosition( unsigned int& Position )
{
  return DichroicMirrorInterface_->GetPosition( Position );
}
bool CDichroicMirror::SetPosition( unsigned int Position )
{
  return DichroicMirrorInterface_->SetPosition( Position );
}
bool CDichroicMirror::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return DichroicMirrorInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterConfigInterface* CDichroicMirror::GetFilterConfigInterface()
{
  return DichroicMirrorInterface_->GetFilterConfigInterface();
}