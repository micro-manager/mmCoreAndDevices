#include "Lens.h"

#include "ASDInterface.h"

CLens::CLens( ILensInterface* LensInterface, int LensIndex, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Lens " + std::to_string( LensIndex ) ),
  LensInterface_( LensInterface ),
  MMDragonfly_( MMDragonfly )
{
  Initialise();
}

CLens::~CLens()
{
}

bool CLens::GetPosition( unsigned int& Position )
{
  return LensInterface_->GetPosition( Position );
}
bool CLens::SetPosition( unsigned int Position )
{
  return LensInterface_->SetPosition( Position );
}
bool CLens::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return LensInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterSet* CLens::GetFilterSet()
{
  return LensInterface_->GetLensConfigInterface();
}