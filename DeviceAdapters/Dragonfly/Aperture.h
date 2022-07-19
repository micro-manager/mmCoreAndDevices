///////////////////////////////////////////////////////////////////////////////
// FILE:          Aperture.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _APERTURE_H_
#define _APERTURE_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include "PositionComponentInterface.h"

class CDragonfly;
class IApertureInterface;

class CAperture : public IPositionComponentInterface
{
public:
  CAperture( IApertureInterface* ApertureInterface, CDragonfly* MMDragonfly );
  ~CAperture();

protected:
  virtual bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* GetFilterSet();

private:
  IApertureInterface* ApertureInterface_;
};

#endif