///////////////////////////////////////////////////////////////////////////////
// FILE:          ILEWrapperInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _ILEWRAPPERINTERFACE_H_
#define _ILEWRAPPERINTERFACE_H_

#include <map>
#include <string>

class IALC_REVObject3;
class IALC_REV_ILEActiveBlankingManagement;
class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;

class IILEWrapperInterface
{
public:

  typedef std::map<std::string, int> TDeviceList;

  virtual void GetListOfDevices( TDeviceList& DeviceList ) = 0;
  virtual bool CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID ) = 0;
  virtual void DeleteILE( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILEActiveBlankingManagement* GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILEPowerManagement* GetILEPowerManagementInterface( IALC_REVObject3 *ILEDevice ) = 0;
};

IILEWrapperInterface* LoadILEWrapper( CIntegratedLaserEngine* Caller);
void UnloadILEWrapper();
#endif
