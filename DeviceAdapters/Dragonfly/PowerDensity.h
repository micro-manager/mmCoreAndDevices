///////////////////////////////////////////////////////////////////////////////
// FILE:          PowerDensity.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _POWERDENSITY_H_
#define _POWERDENSITY_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include "PositionComponentInterface.h"

class CDragonfly;
class IIllLensInterface;

class CPowerDensity : public IPositionComponentInterface
{
public:
  CPowerDensity( IIllLensInterface* IllLensInterface, int LensIndex, CDragonfly* MMDragonfly );
  ~CPowerDensity();

protected:
  virtual bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* GetFilterSet();

private:
  IIllLensInterface* IllLensInterface_;
  CDragonfly* MMDragonfly_;
};

#endif