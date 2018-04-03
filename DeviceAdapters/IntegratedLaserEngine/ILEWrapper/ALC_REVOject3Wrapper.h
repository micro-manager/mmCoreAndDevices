///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REVObject3Wrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REVOBJECT3WRAPPER_H_
#define _ALC_REVOBJECT3WRAPPER_H_

#include "ALC_REV.h"
#include <string>

class CALC_REV_ILEWrapper;
class CALC_REV_Laser2Wrapper;
class CALC_REV_PortWrapper;

class CALC_REVObject3Wrapper : public IALC_REVObject3
{
public:
  CALC_REVObject3Wrapper( HMODULE DLL, const char* UnitID1, const char* UnitID2 = "", bool ILE700 = false );
  CALC_REVObject3Wrapper( IALC_REVObject3* ALC_REVObject3 );
  ~CALC_REVObject3Wrapper();

  IALC_REVObject3* GetILEObject() { return ALC_REVObject3_; }

  // IALC_REVObject3
  IALC_REV_Laser3* __stdcall GetLaserInterface3();
  IALC_REV_ILE* __stdcall GetILEInterface();

  // IALC_REVObject2
  IALC_REV_Laser* __stdcall GetLaserInterface();
  IALC_REV_Laser2* __stdcall GetLaserInterface2();
  IALC_REV_Piezo* __stdcall GetPiezoInterface();
  IALC_REV_DIO* __stdcall GetDIOInterface();
  IALC_REV_Port* __stdcall GetPortInterface();

private:
  HMODULE DLL_;
  IALC_REVObject3* ALC_REVObject3_;
  std::string UnitID1_;
  std::string UnitID2_;
  bool ILE700_;
  bool IsDualILE_;

  CALC_REV_ILEWrapper* ALC_REV_ILEWrapper_;
  CALC_REV_Laser2Wrapper* ALC_REV_Laser2Wrapper_;
  CALC_REV_PortWrapper* ALC_REV_PortWrapper_;

  typedef bool( __stdcall *TCreate_ILE_REV3 )( IALC_REVObject3 **ALC_REVObject3, const char *UnitID );
  typedef bool( __stdcall *TDelete_ILE_REV3 )( IALC_REVObject3 *ALC_REVObject3 );
  typedef bool( __stdcall *TCreate_DUALILE_REV3 )( IALC_REVObject3 **ALC_REVObject3, const char *UnitID1, const char *UnitID2, bool ILE700 );
  typedef bool( __stdcall *TDelete_DUALILE_REV3 )( IALC_REVObject3 *ALC_REVObject3 );

  TCreate_ILE_REV3 Create_ILE_REV3_;
  TDelete_ILE_REV3 Delete_ILE_REV3_;
  TCreate_DUALILE_REV3 Create_DUALILE_REV3_;
  TDelete_DUALILE_REV3 Delete_DUALILE_REV3_;
};

#endif
