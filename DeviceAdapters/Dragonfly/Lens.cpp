#include "Lens.h"
#include "Dragonfly.h"

#include "ASDInterface.h"

CLens::CLens( ILensInterface* LensInterface, int LensIndex, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Imaging Magnification " + std::to_string( static_cast< long long >( LensIndex + 1 ) ), true ),
  LensInterface_( LensInterface ),
  MMDragonfly_(MMDragonfly)
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
  MMDragonfly_->LogComponentMessage( "Set Lens position to [" + std::to_string( Position ) + "]", true );
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