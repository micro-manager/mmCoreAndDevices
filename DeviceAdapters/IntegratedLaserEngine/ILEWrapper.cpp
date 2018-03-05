///////////////////////////////////////////////////////////////////////////////
// FILE:          ILEWrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#ifdef WIN32
#include <windows.h>
#endif

#include "ALC_REV.h"
#include "ILEWrapper.h"
#include "IntegratedLaserEngine.h"
#include "../../MMDevice/DeviceThreads.h"
#include <sstream>
#include <exception>

///////////////////////////////////////////////////////////////////////////////
// ILE Wrapper singleton handling
///////////////////////////////////////////////////////////////////////////////

/** This instance is shared between all ILE devices */
static IILEWrapperInterface* ILEWrapper_s;
MMThreadLock ILEWrapperLock_s;
int CILEWrapper::NbInstances_s = 0;

IILEWrapperInterface* LoadILEWrapper( CIntegratedLaserEngine* Caller )
{
  MMThreadGuard G( ILEWrapperLock_s );
  if ( ILEWrapper_s == nullptr )
  {
    try
    {
      ILEWrapper_s = new CILEWrapper();
    }
    catch ( std::exception &e )
    {
      Caller->LogMMMessage( e.what() );
      throw e;
    }
    CDeviceUtils::SleepMs( 100 );
  }

  ++CILEWrapper::NbInstances_s;
  return ILEWrapper_s;
}

void UnloadILEWrapper()
{
  MMThreadGuard g( ILEWrapperLock_s );
  --CILEWrapper::NbInstances_s;

  if ( CILEWrapper::NbInstances_s == 0 )
  {
    delete ILEWrapper_s;
    ILEWrapper_s = nullptr;
  }
}

///////////////////////////////////////////////////////////////////////////////
// ILE Wrapper
///////////////////////////////////////////////////////////////////////////////

CILEWrapper::CILEWrapper(void) :
  DLL_( nullptr ), 
  Create_ILE_REV3_( nullptr ), 
  Delete_ILE_REV3_( nullptr )
{
#ifdef _M_X64
  std::string libraryName = "AB_ALC_REV64.dll";
#else
  std::string libraryName = "AB_ALC_REV.dll";
#endif
  DLL_ = LoadLibraryA(libraryName.c_str());
  if ( DLL_ == nullptr )
  {
    std::ostringstream vMessage;
    vMessage << "failed to load library: " << libraryName << " check that the library is in your PATH ";
    throw std::runtime_error( vMessage.str() );
  }

  Create_ILE_Detection_ = (TCreate_ILE_Detection)GetProcAddress( DLL_, "Create_ILE_Detection" );
  if ( Create_ILE_Detection_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Create_ILE_Detection failed\n" );
  }
  Delete_ILE_Detection_ = (TDelete_ILE_Detection)GetProcAddress( DLL_, "Delete_ILE_Detection" );
  if ( Delete_ILE_Detection_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Delete_ILE_Detection failed\n" );
  }

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
  
  GetILEActiveBlankingManagementInterface_ = (TGetILEActiveBlankingManagementInterface)GetProcAddress( DLL_, "GetILEActiveBlankingManagementInterface" );
  if ( GetILEActiveBlankingManagementInterface_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress GetILEActiveBlankingManagementInterface_ failed\n" );
  }

  GetILEPowerManagementInterface_ = (TGetILEPowerManagementInterface)GetProcAddress( DLL_, "GetILEPowerManagementInterface" );
  if ( GetILEPowerManagementInterface_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress GetILEPowerManagementInterface_ failed\n" );
  }

  Create_ILE_Detection_( &ILEDetection_ );
  if ( ILEDetection_ == nullptr )
  {
    throw std::runtime_error( "Create_ILE_Detection failed" );
  }
}

CILEWrapper::~CILEWrapper()
{
  if ( ILEDetection_ != nullptr )
  {
    Delete_ILE_Detection_( ILEDetection_ );
  }
  ILEDetection_ = nullptr;

  if ( DLL_ != nullptr )
  {
    FreeLibrary( DLL_ );
  }
  DLL_ = nullptr;
}

void CILEWrapper::GetListOfDevices( TDeviceList& DeviceList )
{
  char vSerialNumber[64];
  int vNumberDevices = ILEDetection_->GetNumberOfDevices();
  DeviceList["Demo"] = 0;
  for (int vDeviceIndex = 1; vDeviceIndex < vNumberDevices + 1; vDeviceIndex++ )
  {
    if ( ILEDetection_->GetSerialNumber( vDeviceIndex, vSerialNumber, 64 ) )
    {
      DeviceList[vSerialNumber] = vDeviceIndex;
    }
  }
}

bool CILEWrapper::CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID )
{
  return Create_ILE_REV3_( ILEDevice, UnitID );
}

void CILEWrapper::DeleteILE( IALC_REVObject3 *ILEDevice )
{
  Delete_ILE_REV3_( ILEDevice );
}

IALC_REV_ILEActiveBlankingManagement* CILEWrapper::GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice )
{
  return GetILEActiveBlankingManagementInterface_( ILEDevice );
}

IALC_REV_ILEPowerManagement* CILEWrapper::GetILEPowerManagementInterface( IALC_REVObject3 *ILEDevice )
{
  return GetILEPowerManagementInterface_( ILEDevice );
}