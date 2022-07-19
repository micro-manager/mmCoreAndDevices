///////////////////////////////////////////////////////////////////////////////
// FILE:          FilterWheelDeviceInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _FILTERWHEELDEVICEINTERFACE_H_
#define _FILTERWHEELDEVICEINTERFACE_H_

class IFilterConfigInterface;

class IFilterWheelDeviceInterface
{
public:
  virtual bool GetPosition( unsigned int& Position ) = 0;
  virtual bool SetPosition( unsigned int Position ) = 0;
  virtual bool GetLimits( unsigned int& MinPosition, unsigned int&MaxPosition ) = 0;
  virtual IFilterConfigInterface* GetFilterConfigInterface() = 0;
  virtual std::string ParseDescription( const std::string& Description ) { return Description; };
};
#endif