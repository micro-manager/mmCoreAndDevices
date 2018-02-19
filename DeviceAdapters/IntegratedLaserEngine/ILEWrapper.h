///////////////////////////////////////////////////////////////////////////////
// FILE:          ILEWrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _ILEWRAPPER_H_
#define _ILEWRAPPER_H_

#include "ILEWrapperInterface.h"

class CILEWrapper : public IILEWrapperInterface
{
public:
  static int NbInstances_s;

  CILEWrapper();
  ~CILEWrapper();

  void GetListOfDevices( TDeviceList& DeviceList );
  bool CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID );
  void DeleteILE( IALC_REVObject3 *ILEDevice );
  IALC_REV_ILEActiveBlankingManagement* GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice );

private:
  HMODULE DLL_;
  typedef bool( __stdcall *TCreate_ILE_Detection )( IILE_Detection **ILEDetection );
  typedef bool( __stdcall *TDelete_ILE_Detection )( IILE_Detection *ILEDetection );
  typedef bool( __stdcall *TCreate_ILE_REV3 )( IALC_REVObject3 **ALC_REVObject3, const char *UnitID );
  typedef bool( __stdcall *TDelete_ILE_REV3 )( IALC_REVObject3 *ALC_REVObject3 );
  typedef IALC_REV_ILEActiveBlankingManagement* ( __stdcall *TGetILEActiveBlankingManagementInterface )( IALC_REVObject3 *ALC_REVObject3 );

  TCreate_ILE_Detection Create_ILE_Detection_;
  TDelete_ILE_Detection Delete_ILE_Detection_;
  TCreate_ILE_REV3 Create_ILE_REV3_;
  TDelete_ILE_REV3 Delete_ILE_REV3_;
  TGetILEActiveBlankingManagementInterface GetILEActiveBlankingManagementInterface_;

  IILE_Detection *ILEDetection_;
};

#endif
