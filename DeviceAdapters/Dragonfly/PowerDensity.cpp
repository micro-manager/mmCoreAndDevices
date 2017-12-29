#include "PowerDensity.h"

#include "ASDInterface.h"

CPowerDensity::CPowerDensity( IIllLensInterface* IllLensInterface, int LensIndex, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Power Density " + std::to_string( LensIndex ) ),
  IllLensInterface_( IllLensInterface ),
  MMDragonfly_( MMDragonfly )
{
  Initialise();
}

CPowerDensity::~CPowerDensity()
{
}

bool CPowerDensity::GetPosition( unsigned int& Position )
{
  return IllLensInterface_->GetPosition( Position );
}
bool CPowerDensity::SetPosition( unsigned int Position )
{
  return IllLensInterface_->SetPosition( Position );
}
bool CPowerDensity::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return IllLensInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterSet* CPowerDensity::GetFilterSet()
{
  return IllLensInterface_->GetLensConfigInterface();
}