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

class CILEWrapper
{
public:
  IILE_Detection *ILEDetection_;
  IALC_REVObject3 *ALC_REVObject3_;

  IALC_REV_Laser2 *pALC_REVLaser2_;
  IALC_REV_DIO *pALC_REV_DIO_;

  CILEWrapper();
  ~CILEWrapper();

private:
  HMODULE alcHandle_;
  typedef bool( __stdcall *TCreate_ILE_Detection )( IILE_Detection **ILEDetection );
  typedef bool( __stdcall *TDelete_ILE_Detection )( IILE_Detection *ILEDetection );
  typedef bool( __stdcall *TCreate_ILE_REV3 )( IALC_REVObject3 **ALC_REVObject3, const char *UnitID );
  typedef bool( __stdcall *TDelete_ILE_REV3 )( IALC_REVObject3 *ALC_REVObject3 );

  TCreate_ILE_Detection Create_ILE_Detection_;
  TDelete_ILE_Detection Delete_ILE_Detection_;
  TCreate_ILE_REV3 Create_ILE_REV3_;
  TDelete_ILE_REV3 Delete_ILE_REV3_;
};

#endif
