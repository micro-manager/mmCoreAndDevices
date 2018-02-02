#include "Aperture.h"

#include "ASDInterface.h"

CAperture::CAperture( IApertureInterface* ApertureInterface, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Field Aperture", true ),
  ApertureInterface_( ApertureInterface )
{
  Initialise();
}

CAperture::~CAperture()
{
}

bool CAperture::GetPosition( unsigned int& Position )
{
  return ApertureInterface_->GetPosition( Position );
}
bool CAperture::SetPosition( unsigned int Position )
{
  return ApertureInterface_->SetPosition( Position );
}
bool CAperture::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return ApertureInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterSet* CAperture::GetFilterSet()
{
  return ApertureInterface_->GetApertureConfigInterface();
}