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

#include <vector>
#include <string>

class IALC_REVObject3;
class IALC_REV_ILEActiveBlankingManagement;
class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;
class IALC_REV_ILE2;
class IALC_REV_ILE4;

class IILEWrapperInterface
{
public:

  typedef std::vector<std::string> TDeviceList;

  virtual ~IILEWrapperInterface() = default;

  virtual void GetListOfDevices( TDeviceList& DeviceList ) = 0;
  virtual bool CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID ) = 0;
  virtual void DeleteILE( IALC_REVObject3 *ILEDevice ) = 0;
  virtual bool CreateDualILE( IALC_REVObject3 **ILEDevice, const char *UnitID1, const char *UnitID2, bool ILE700 ) = 0;
  virtual void DeleteDualILE( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILEActiveBlankingManagement* GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILEPowerManagement* GetILEPowerManagementInterface( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILE2* GetILEInterface2( IALC_REVObject3 *ILEDevice ) = 0;
  virtual IALC_REV_ILE4* GetILEInterface4( IALC_REVObject3 *ILEDevice ) = 0;
};

IILEWrapperInterface* LoadILEWrapper( CIntegratedLaserEngine* Caller);
void UnloadILEWrapper();
#endif
