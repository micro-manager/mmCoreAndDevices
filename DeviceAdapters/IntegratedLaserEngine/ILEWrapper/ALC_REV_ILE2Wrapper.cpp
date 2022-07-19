///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILE2Wrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifdef WIN32
#include <windows.h>
#endif
#include "ALC_REV_ILE2Wrapper.h"
#include "ILESDKLock.h"
#include "ALC_REVOject3Wrapper.h"
#include <stdexcept>


CALC_REV_ILE2Wrapper::CALC_REV_ILE2Wrapper( IALC_REV_ILE2* ALC_REV_ILE2 ) :
  ALC_REV_ILE2_( ALC_REV_ILE2 ),
  ILEDevice1_( nullptr ),
  ILEDevice2_( nullptr )
{
  if ( ALC_REV_ILE2_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILE2 pointer passed to CALC_REV_ILE2Wrapper is null" );
  }
}

CALC_REV_ILE2Wrapper::~CALC_REV_ILE2Wrapper()
{
  delete ILEDevice1_;
  delete ILEDevice2_;
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILE2
///////////////////////////////////////////////////////////////////////////////

bool CALC_REV_ILE2Wrapper::GetNumberOfLasers( int *NumLasersUnit1, int *NumLasersUnit2 )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->GetNumberOfLasers( NumLasersUnit1, NumLasersUnit2 );
}

bool CALC_REV_ILE2Wrapper::GetNumberOfPorts( int *NumPortsUnit1, int *NumPortsUnit2 )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->GetNumberOfPorts( NumPortsUnit1, NumPortsUnit2 );
}

bool CALC_REV_ILE2Wrapper::GetPortIndex( int *PortIndexUnit1, int *PortIndexUnit2 )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->GetPortIndex( PortIndexUnit1, PortIndexUnit2 );
}

bool CALC_REV_ILE2Wrapper::SetPortIndex( int PortIndexUnit1, int PortIndexUnit2 )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->SetPortIndex( PortIndexUnit1, PortIndexUnit2 );
}

bool CALC_REV_ILE2Wrapper::CalculatePort( int Port1, int Port2, int *Port )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->CalculatePort( Port1, Port2, Port );
}

bool CALC_REV_ILE2Wrapper::ExtractPort( int Port, int *Port1, int *Port2 )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->ExtractPort( Port, Port1, Port2 );
}

bool CALC_REV_ILE2Wrapper::GetInterface( IALC_REVObject3 **ILE1, IALC_REVObject3 **ILE2 )
{
  bool vRet = false;
  if ( ILEDevice1_ == nullptr && ILEDevice2_ == nullptr )
  {
    IALC_REVObject3 *vILEDevice1, *vILEDevice2;
    {
      CILESDKLock vSDKLock;
      vRet = ALC_REV_ILE2_->GetInterface( &vILEDevice1, &vILEDevice2 );
    }
    if ( vRet )
    {
      ILEDevice1_ = new CALC_REVObject3Wrapper( vILEDevice1 );
      ILEDevice2_ = new CALC_REVObject3Wrapper( vILEDevice2 );
      *ILE1 = ILEDevice1_;
      *ILE2 = ILEDevice2_;
    }
  }
  else
  {
    *ILE1 = ILEDevice1_;
    *ILE2 = ILEDevice2_;
    vRet = true;
  }
  return vRet;
}

bool CALC_REV_ILE2Wrapper::IsILE700()
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->IsILE700();
}

bool CALC_REV_ILE2Wrapper::GetCurrentPowerIntoFiberForDualUnit( double *Power1_mW, double *Power2_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->GetCurrentPowerIntoFiberForDualUnit( Power1_mW, Power2_mW );
}

bool CALC_REV_ILE2Wrapper::AdjustPowerIntoInputFibre()
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE2_->AdjustPowerIntoInputFibre();
}
