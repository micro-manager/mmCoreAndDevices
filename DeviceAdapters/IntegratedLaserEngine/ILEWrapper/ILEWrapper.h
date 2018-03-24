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

#include "..\ILEWrapperInterface.h"
#include <map>

class CALC_REV_ILEActiveBlankingManagementWrapper;
class CALC_REV_ILEPowerManagementWrapper;
class CALC_REV_ILE2Wrapper;
class CALC_REV_ILE4Wrapper;

class CILEWrapper : public IILEWrapperInterface
{
public:
  static int NbInstances_s;

  CILEWrapper();
  ~CILEWrapper();

  void GetListOfDevices( TDeviceList& DeviceList );
  bool CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID );
  void DeleteILE( IALC_REVObject3 *ILEDevice );
  bool CreateDualILE( IALC_REVObject3 **ILEDevice, const char *UnitID1, const char *UnitID2, bool ILE700 );
  void DeleteDualILE( IALC_REVObject3 *ILEDevice );
  IALC_REV_ILEActiveBlankingManagement* GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice );
  IALC_REV_ILEPowerManagement* GetILEPowerManagementInterface( IALC_REVObject3 *ILEDevice );
  IALC_REV_ILE2* GetILEInterface2( IALC_REVObject3 *ILEDevice );
  IALC_REV_ILE4* GetILEInterface4( IALC_REVObject3 *ILEDevice );

private:
  HMODULE DLL_;
  typedef bool( __stdcall *TCreate_ILE_Detection )( IILE_Detection **ILEDetection );
  typedef bool( __stdcall *TDelete_ILE_Detection )( IILE_Detection *ILEDetection );
  typedef IALC_REV_ILEActiveBlankingManagement* ( __stdcall *TGetILEActiveBlankingManagementInterface )( IALC_REVObject3 *ALC_REVObject3 );
  typedef IALC_REV_ILEPowerManagement* ( __stdcall *TGetILEPowerManagementInterface )( IALC_REVObject3 *ALC_REVObject3 );
  typedef IALC_REV_ILE2* ( __stdcall *TGetILEInterface2 )( IALC_REVObject3 *ILEDevice );
  typedef IALC_REV_ILE4* ( __stdcall *TGetILEInterface4 )( IALC_REVObject3 *ILEDevice );

  TCreate_ILE_Detection Create_ILE_Detection_;
  TDelete_ILE_Detection Delete_ILE_Detection_;
  TGetILEActiveBlankingManagementInterface GetILEActiveBlankingManagementInterface_;
  TGetILEPowerManagementInterface GetILEPowerManagementInterface_;
  TGetILEInterface2 GetILEInterface2_;
  TGetILEInterface4 GetILEInterface4_;

  IILE_Detection *ILEDetection_;
  typedef std::map<IALC_REV_ILEActiveBlankingManagement*, CALC_REV_ILEActiveBlankingManagementWrapper*> TActiveBlankingManagementMap;
  TActiveBlankingManagementMap ActiveBlankingManagementMap_;
  typedef std::map<IALC_REV_ILEPowerManagement*, CALC_REV_ILEPowerManagementWrapper*> TPowerManagementMap;
  TPowerManagementMap PowerManagementMap_;
  typedef std::map<IALC_REV_ILE2*, CALC_REV_ILE2Wrapper*> TILE2Map;
  TILE2Map ILE2Map_;
  typedef std::map<IALC_REV_ILE4*, CALC_REV_ILE4Wrapper*> TILE4Map;
  TILE4Map ILE4Map_;
};

#endif
