#include "PowerDensity.h"
#include "Dragonfly.h"

#include "ASDInterface.h"

class TPowerDensityNotification : public INotify
{
private:
  CPowerDensity *Owner_;
public:
  TPowerDensityNotification( CPowerDensity *Owner )
    :Owner_( Owner )
  {
  }
  void __stdcall Notify()
  {
    Owner_->RestrictionNotification();
  };
};

CPowerDensity::CPowerDensity( IIllLensInterface* IllLensInterface, int LensIndex, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, (LensIndex == 0) ? "Power Density" : "Power Density " + std::to_string( LensIndex + 1 ) ),
  IllLensInterface_( IllLensInterface ),
  RestrictionStatusChangeNotified_( true ),
  RestrictionNotification_( nullptr )
{
  Initialise();
  RestrictionNotification_ = new TPowerDensityNotification( this );
  IllLensInterface_->RegisterForNotificationOnRangeRestriction( RestrictionNotification_ );
}

CPowerDensity::~CPowerDensity()
{
  delete RestrictionNotification_;
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

void CPowerDensity::RestrictionNotification()
{
  RestrictionStatusChangeNotified_ = true;
}

bool CPowerDensity::UpdateAllowedValues()
{
  if ( !RestrictionStatusChangeNotified_ ) return false;
  RestrictionStatusChangeNotified_ = false;

  if ( IllLensInterface_->IsRestrictionEnabled() )
  {
    unsigned int vMinPosition;
    unsigned int vMaxPosition;
    if ( IllLensInterface_->GetRestrictedRange( vMinPosition, vMaxPosition ) )
    {
      MMDragonfly_->ClearAllowedValues( PropertyName_.c_str() );
      const TPositionNameMap& vPositionNameMap = GetPositionNameMap();
      for ( unsigned int vPositionIndex = vMinPosition; vPositionIndex <= vMaxPosition; ++vPositionIndex )
      {
        TPositionNameMap::const_iterator vPosition = vPositionNameMap.find( vPositionIndex );
        if ( vPosition != vPositionNameMap.end() )
        {
          MMDragonfly_->AddAllowedValue( PropertyName_.c_str(), vPosition->second.c_str() );
        }
      }
    }
  }
  else
  {
    MMDragonfly_->ClearAllowedValues( PropertyName_.c_str() );
    const TPositionNameMap& vPositionNameMap = GetPositionNameMap();
    TPositionNameMap::const_iterator vPositionIt = vPositionNameMap.begin();
    while ( vPositionIt != vPositionNameMap.end() )
    {
      MMDragonfly_->AddAllowedValue( PropertyName_.c_str(), vPositionIt->second.c_str() );
      vPositionIt++;
    }
  }
  return true;
}