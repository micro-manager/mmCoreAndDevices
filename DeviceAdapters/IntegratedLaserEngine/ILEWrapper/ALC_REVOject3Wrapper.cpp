///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REVObject3Wrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifdef WIN32
#include <windows.h>
#endif

#include "ALC_REVOject3Wrapper.h"
#include "ALC_REV_ILEWrapper.h"
#include "ALC_REV_Laser2Wrapper.h"
#include "ALC_REV_PortWrapper.h"
#include "ILESDKLock.h"
#include <exception>


CALC_REVObject3Wrapper::CALC_REVObject3Wrapper( HMODULE DLL, const char* UnitID ) :
  DLL_( DLL ),
  Create_ILE_REV3_( nullptr ),
  Delete_ILE_REV3_( nullptr ),
  ALC_REV_ILEWrapper_( nullptr ),
  ALC_REV_Laser2Wrapper_( nullptr ),
  ALC_REV_PortWrapper_( nullptr ),
  ALC_REVObject3_( nullptr )
{
  UnitID_ = UnitID;

  Create_ILE_REV3_ = (TCreate_ILE_REV3)GetProcAddress( DLL_, "Create_ILE_REV3" );
  if( Create_ILE_REV3_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Create_ILE_REV3 failed\n" );
  }
  Delete_ILE_REV3_ = (TDelete_ILE_REV3)GetProcAddress( DLL_, "Delete_ILE_REV3" );
  if( Delete_ILE_REV3_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Delete_ILE_REV3 failed\n" );
  }

  CILESDKLock vSDKLock;
  bool vRet = Create_ILE_REV3_( &ALC_REVObject3_, UnitID );
  if ( !vRet )
  {
    throw std::runtime_error( "Create_ILE_REV3 failed" );
  }
}

CALC_REVObject3Wrapper::~CALC_REVObject3Wrapper()
{
  delete ALC_REV_ILEWrapper_;
  delete ALC_REV_Laser2Wrapper_;
  delete ALC_REV_PortWrapper_;
  CILESDKLock vSDKLock;
  Delete_ILE_REV3_( ALC_REVObject3_ );
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REVObject3
///////////////////////////////////////////////////////////////////////////////

IALC_REV_Laser3* CALC_REVObject3Wrapper::GetLaserInterface3()
{
  throw std::logic_error( "CALC_REVObject3Wrapper::GetLaserInterface3() not implemented" );
}

IALC_REV_ILE* CALC_REVObject3Wrapper::GetILEInterface()
{
  if ( ALC_REV_ILEWrapper_ == nullptr )
  {
    CILESDKLock vSDKLock;
    IALC_REV_ILE* vALC_REV_ILE = ALC_REVObject3_->GetILEInterface();
    if ( vALC_REV_ILE != nullptr )
    {
      ALC_REV_ILEWrapper_ = new CALC_REV_ILEWrapper( vALC_REV_ILE );
    }
  }
  return ALC_REV_ILEWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REVObject2
///////////////////////////////////////////////////////////////////////////////

IALC_REV_Laser* CALC_REVObject3Wrapper::GetLaserInterface()
{
  throw std::logic_error( "CALC_REVObject3Wrapper::GetLaserInterface() not implemented" );
}

IALC_REV_Laser2* CALC_REVObject3Wrapper::GetLaserInterface2()
{
  if ( ALC_REV_Laser2Wrapper_ == nullptr )
  {
    CILESDKLock vSDKLock;
    IALC_REV_Laser2* vALC_REV_Laser2 = ALC_REVObject3_->GetLaserInterface2();
    if ( vALC_REV_Laser2 != nullptr )
    {
      ALC_REV_Laser2Wrapper_ = new CALC_REV_Laser2Wrapper( vALC_REV_Laser2 );
    }
  }
  return ALC_REV_Laser2Wrapper_;
}

IALC_REV_Piezo* CALC_REVObject3Wrapper::GetPiezoInterface()
{
  throw std::logic_error( "CALC_REVObject3Wrapper::GetPiezoInterface() not implemented" );
}

IALC_REV_DIO* CALC_REVObject3Wrapper::GetDIOInterface()
{
  throw std::logic_error( "CALC_REVObject3Wrapper::GetDIOInterface() not implemented" );
}

IALC_REV_Port* CALC_REVObject3Wrapper::GetPortInterface()
{
  if ( ALC_REV_PortWrapper_ == nullptr )
  {
    CILESDKLock vSDKLock;
    IALC_REV_Port* vALC_REV_Port = ALC_REVObject3_->GetPortInterface();
    if ( vALC_REV_Port != nullptr )
    {
      ALC_REV_PortWrapper_ = new CALC_REV_PortWrapper( vALC_REV_Port );
    }
  }
  return ALC_REV_PortWrapper_;
}