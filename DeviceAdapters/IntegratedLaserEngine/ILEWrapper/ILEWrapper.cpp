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
#include "ALC_REVOject3Wrapper.h"
#include "ALC_REV_ILEActiveBlankingManagementWrapper.h"
#include "ALC_REV_ILEPowerManagementWrapper.h"
#include "ALC_REV_ILE2Wrapper.h"
#include "ALC_REV_ILE4Wrapper.h"
#include "ILESDKLock.h"
#include "../IntegratedLaserEngine.h"
#include "../../../MMDevice/DeviceThreads.h"
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

CILEWrapper::CILEWrapper() :
  DLL_( nullptr ),
  Create_ILE_Detection_( nullptr ),
  Delete_ILE_Detection_( nullptr ),
  GetILEActiveBlankingManagementInterface_( nullptr ),
  GetILEPowerManagementInterface_( nullptr ),
  GetILEInterface2_( nullptr ),
  GetILEInterface4_( nullptr )
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

  GetILEInterface2_ = (TGetILEInterface2)GetProcAddress( DLL_, "GetILEInterface2" );
  if ( GetILEInterface2_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress GetILEInterface2_ failed\n" );
  }

  GetILEInterface4_ = (TGetILEInterface4)GetProcAddress( DLL_, "GetILEInterface4" );
  if ( GetILEInterface4_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress GetILEInterface4_ failed\n" );
  }

  Create_ILE_Detection_( &ILEDetection_ );
  if ( ILEDetection_ == nullptr )
  {
    throw std::runtime_error( "Create_ILE_Detection failed" );
  }
}

CILEWrapper::~CILEWrapper()
{
  for ( auto& elt : ActiveBlankingManagementMap_ )
  {
    delete elt.second;
  }

  for ( auto& elt : PowerManagementMap_ )
  {
    delete elt.second;
  }

  for ( auto& elt : ILE2Map_ )
  {
    delete elt.second;
  }

  for ( auto& elt : ILE4Map_ )
  {
    delete elt.second;
  }

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
  CILESDKLock vSDKLock;
  char vSerialNumber[64];
  int vNumberDevices = ILEDetection_->GetNumberOfDevices();
  DeviceList.push_back("Demo");
  DeviceList.push_back( "DemoILE701" );
  DeviceList.push_back( "DemoILE702" );
  DeviceList.push_back( "DemoILE703 " );
  DeviceList.push_back( "DemoILE704" );
  for (int vDeviceIndex = 0; vDeviceIndex < vNumberDevices; vDeviceIndex++ )
  {
    if ( ILEDetection_->GetSerialNumber( vDeviceIndex, vSerialNumber, 64 ) )
    {
      DeviceList.push_back(vSerialNumber);
    }
  }
}

bool CILEWrapper::CreateILE( IALC_REVObject3 **ILEDevice, const char *UnitID )
{
  bool vRet = false;
  try
  {
    CALC_REVObject3Wrapper* vALC_REVObject3Wrapper = new CALC_REVObject3Wrapper( DLL_, UnitID );
    *ILEDevice = vALC_REVObject3Wrapper;
    vRet = true;
  }
  catch ( std::exception& )
  {
  }
  return vRet;
}

bool CILEWrapper::CreateDualILE( IALC_REVObject3 **ILEDevice, const char *UnitID1, const char *UnitID2, bool ILE700 )
{
  bool vRet = false;
  try
  {
    CALC_REVObject3Wrapper* vALC_REVObject3Wrapper = new CALC_REVObject3Wrapper( DLL_, UnitID1, UnitID2, ILE700 );
    *ILEDevice = vALC_REVObject3Wrapper;
    vRet = true;
  }
  catch ( std::exception& )
  {
  }
  return vRet;
}

void CILEWrapper::DeleteILE( IALC_REVObject3 *ILEDevice )
{
  CALC_REVObject3Wrapper* vALC_REVObject3Wrapper = dynamic_cast<CALC_REVObject3Wrapper*>( ILEDevice );
  if ( vALC_REVObject3Wrapper != nullptr )
  {
    delete vALC_REVObject3Wrapper;
  }
}

void CILEWrapper::DeleteDualILE( IALC_REVObject3 *ILEDevice )
{
  // Just a placeholder in case we need to split the ALC_RevObject3 wrapper into single and dual ILE
  DeleteILE( ILEDevice );
}

IALC_REV_ILEActiveBlankingManagement* CILEWrapper::GetILEActiveBlankingManagementInterface( IALC_REVObject3 *ILEDevice )
{
  CALC_REV_ILEActiveBlankingManagementWrapper* vActiveBlankingManagementWrapper = nullptr;
  CALC_REVObject3Wrapper* vALC_REVObjectWrapper = dynamic_cast<CALC_REVObject3Wrapper*>( ILEDevice );
  if ( vALC_REVObjectWrapper != nullptr )
  {
    IALC_REVObject3* vILEObject = vALC_REVObjectWrapper->GetILEObject();
    IALC_REV_ILEActiveBlankingManagement* vILEActiveBlankingManagement = nullptr;
    {
      CILESDKLock vSDKLock;
      vILEActiveBlankingManagement = GetILEActiveBlankingManagementInterface_( vILEObject );
    }
    if ( vILEActiveBlankingManagement != nullptr )
    {
      TActiveBlankingManagementMap::iterator vActiveBlankingIt = ActiveBlankingManagementMap_.find( vILEActiveBlankingManagement );
      if ( vActiveBlankingIt != ActiveBlankingManagementMap_.end() )
      {
        // Prevent creating a wrapper object every time this function is called
        // We could assume that for a given instance of IALC_REVObject3 the same pointer to active blanking would always be returned
        // But if that behaviour was to change in the DLL then a more optimised solution would break
        vActiveBlankingManagementWrapper = vActiveBlankingIt->second;
      }
      else
      {
        vActiveBlankingManagementWrapper = new CALC_REV_ILEActiveBlankingManagementWrapper( vILEActiveBlankingManagement );
        ActiveBlankingManagementMap_[vILEActiveBlankingManagement] = vActiveBlankingManagementWrapper;
      }
    }
  }
  return vActiveBlankingManagementWrapper;
}

IALC_REV_ILEPowerManagement* CILEWrapper::GetILEPowerManagementInterface( IALC_REVObject3 *ILEDevice )
{
  CALC_REV_ILEPowerManagementWrapper* vPowerManagementWrapper = nullptr;
  CALC_REVObject3Wrapper* vALC_REVObjectWrapper = dynamic_cast<CALC_REVObject3Wrapper*>( ILEDevice );
  if ( vALC_REVObjectWrapper != nullptr )
  {
    IALC_REVObject3* vILEObject = vALC_REVObjectWrapper->GetILEObject();
    IALC_REV_ILEPowerManagement* vILEPowerManagement = nullptr;
    {
      CILESDKLock vSDKLock;
      vILEPowerManagement = GetILEPowerManagementInterface_( vILEObject );
    }
    if ( vILEPowerManagement != nullptr )
    {
      TPowerManagementMap::iterator vPowerManagementIt = PowerManagementMap_.find( vILEPowerManagement );
      if ( vPowerManagementIt != PowerManagementMap_.end() )
      {
        // Prevent creating a wrapper object every time this function is called
        // We could assume that for a given instance of IALC_REVObject3 the same pointer to power management would always be returned
        // But if that behaviour was to change in the DLL then a more optimised solution would break
        vPowerManagementWrapper = vPowerManagementIt->second;
      }
      else
      {
        vPowerManagementWrapper = new CALC_REV_ILEPowerManagementWrapper( vILEPowerManagement );
        PowerManagementMap_[vILEPowerManagement] = vPowerManagementWrapper;
      }
    }
  }
  return vPowerManagementWrapper;
}

IALC_REV_ILE2* CILEWrapper::GetILEInterface2( IALC_REVObject3 *ILEDevice )
{
  CALC_REV_ILE2Wrapper* vILE2Wrapper = nullptr;
  CALC_REVObject3Wrapper* vALC_REVObjectWrapper = dynamic_cast<CALC_REVObject3Wrapper*>( ILEDevice );
  if ( vALC_REVObjectWrapper != nullptr )
  {
    IALC_REVObject3* vILEObject = vALC_REVObjectWrapper->GetILEObject();
    IALC_REV_ILE2* vILE2 = nullptr;
    {
      CILESDKLock vSDKLock;
      vILE2 = GetILEInterface2_( vILEObject );
    }
    if ( vILE2 != nullptr )
    {
      TILE2Map::iterator vILE2It = ILE2Map_.find( vILE2 );
      if ( vILE2It != ILE2Map_.end() )
      {
        // Prevent creating a wrapper object every time this function is called
        // We could assume that for a given instance of IALC_REVObject3 the same pointer to ILE Interface 2 would always be returned
        // But if that behaviour was to change in the DLL then a more optimised solution would break
        vILE2Wrapper = vILE2It->second;
      }
      else
      {
        vILE2Wrapper = new CALC_REV_ILE2Wrapper( vILE2 );
        ILE2Map_[vILE2] = vILE2Wrapper;
      }
    }
  }
  return vILE2Wrapper;
}

IALC_REV_ILE4* CILEWrapper::GetILEInterface4( IALC_REVObject3 *ILEDevice )
{
  CALC_REV_ILE4Wrapper* vILE4Wrapper = nullptr;
  CALC_REVObject3Wrapper* vALC_REVObjectWrapper = dynamic_cast<CALC_REVObject3Wrapper*>( ILEDevice );
  if ( vALC_REVObjectWrapper != nullptr )
  {
    IALC_REVObject3* vILEObject = vALC_REVObjectWrapper->GetILEObject();
    IALC_REV_ILE4* vILE4 = nullptr;
    {
      CILESDKLock vSDKLock;
      vILE4 = GetILEInterface4_( vILEObject );
    }
    if ( vILE4 != nullptr )
    {
      TILE4Map::iterator vILE4It = ILE4Map_.find( vILE4 );
      if ( vILE4It != ILE4Map_.end() )
      {
        // Prevent creating a wrapper object every time this function is called
        // We could assume that for a given instance of IALC_REVObject3 the same pointer to ILE Interface 4 would always be returned
        // But if that behaviour was to change in the DLL then a more optimised solution would break
        vILE4Wrapper = vILE4It->second;
      }
      else
      {
        vILE4Wrapper = new CALC_REV_ILE4Wrapper( vILE4 );
        ILE4Map_[vILE4] = vILE4Wrapper;
      }
    }
  }
  return vILE4Wrapper;
}
