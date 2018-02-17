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
#include <string>
#include <sstream>
#include <exception>


CILEWrapper::CILEWrapper(void)
  : alcHandle_( nullptr ), 
  Create_ILE_REV3_( nullptr ), 
  Delete_ILE_REV3_( nullptr ), 
  ALC_REVObject3_( nullptr ),
  pALC_REVLaser2_( nullptr ),
  pALC_REV_DIO_( nullptr )
{
#ifdef _M_X64
  std::string libraryName = "AB_ALC_REV64.dll";
#else
  std::string libraryName = "AB_ALC_REV.dll";
#endif
  alcHandle_ = LoadLibraryA(libraryName.c_str());
  if ( alcHandle_ == nullptr )
  {
      std::ostringstream messs;
      messs << "failed to load library: " << libraryName << " check that the library is in your PATH ";
      throw std::runtime_error( messs.str() );
  }

  Create_ILE_Detection_ = (TCreate_ILE_Detection)GetProcAddress( alcHandle_, "Create_ILE_Detection" );
  if ( Create_ILE_Detection_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Create_ILE_Detection failed\n" );
  }
  Delete_ILE_Detection_ = (TDelete_ILE_Detection)GetProcAddress( alcHandle_, "Delete_ILE_Detection" );
  if ( Delete_ILE_Detection_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Delete_ILE_Detection failed\n" );
  }

  Create_ILE_REV3_ = (TCreate_ILE_REV3)GetProcAddress( alcHandle_, "Create_ILE_REV3" );
  if( Create_ILE_REV3_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Create_ILE_REV3 failed\n" );
  }
  Delete_ILE_REV3_ = (TDelete_ILE_REV3)GetProcAddress( alcHandle_, "Delete_ILE_REV3" );
  if( Delete_ILE_REV3_ == nullptr )
  {
    throw std::runtime_error( "GetProcAddress Delete_ILE_REV3 failed\n" );
  }

  Create_ILE_Detection_( &ILEDetection_ );
  if ( ILEDetection_ == nullptr )
  {
    throw std::runtime_error( "Create_ILE_Detection failed" );
  }
  char vSerialNumber[64];
  int vNumberDevices = ILEDetection_->GetNumberOfDevices();
  if ( vNumberDevices < 0 )
  {
    //TODO: Throw?
    throw std::runtime_error( "No device found" );
  }
  else
  {
    //TODO: retrieve the serial numbers of all devices to display them to the user and let him choose which one to connect to
    //=> has to be done in a pre-init property
    ILEDetection_->GetSerialNumber( 1, vSerialNumber, 64 );
  }

  Create_ILE_REV3_( &ALC_REVObject3_, vSerialNumber );
  if ( ALC_REVObject3_ == nullptr )
  {
    throw std::runtime_error( "Create_ILE_REV3 failed" );
  }

  //TODO: remove those 2
  pALC_REVLaser2_ = ALC_REVObject3_->GetLaserInterface2( );
  if ( pALC_REVLaser2_ == nullptr )
    throw std::runtime_error( "GetLaserInterface failed" );

  //pALC_REV_DIO_ = ALC_REVObject3_->GetDIOInterface( );
  //if ( pALC_REV_DIO_ == nullptr )
  //  throw std::runtime_error( " GetDIOInterface failed!" );
}

CILEWrapper::~CILEWrapper()
{
  if( ALC_REVObject3_ != nullptr )
    Delete_ILE_REV3_( ALC_REVObject3_ );
  ALC_REVObject3_ = nullptr;

  if( alcHandle_ != nullptr )
      FreeLibrary(alcHandle_);
  alcHandle_ = nullptr;
}
