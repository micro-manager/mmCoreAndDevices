///////////////////////////////////////////////////////////////////////////////
// FILE:          Lens.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _LENS_H_
#define _LENS_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include "PositionComponentInterface.h"

class ILensInterface;

class CLens : public IPositionComponentInterface
{
public:
  CLens( ILensInterface* LensInterface, int LensIndex, CDragonfly* MMDragonfly );
  ~CLens();

protected:
  virtual bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* GetFilterSet();

private:
  ILensInterface* LensInterface_;
};

#endif