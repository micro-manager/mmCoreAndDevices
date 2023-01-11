#include "Aperture.h"
#include "Dragonfly.h"

#include "ASDInterface.h"

CAperture::CAperture( IApertureInterface* ApertureInterface, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Field Aperture", true ),
  ApertureInterface_( ApertureInterface ),
  MMDragonfly_(MMDragonfly)
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
  MMDragonfly_->LogComponentMessage( "Set Aperture position to [" + std::to_string( Position ) + "]", true );
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