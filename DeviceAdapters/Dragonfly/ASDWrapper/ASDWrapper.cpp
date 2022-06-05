#include "ASDWrapper.h"
#include "ASDWrapperLoader.h"
#include "ASDWrapperInterface.h"
#include <stdexcept>

CASDWrapper::CASDWrapper()
  : DLL_( nullptr ),
  mCreateASDLoader( nullptr ),
  mDeleteASDLoader( nullptr ),
  ASDWrapperInterface4_( nullptr ),
  ASDWrapperInterface6_( nullptr )
{
#ifdef _M_X64
  DLL_ = LoadLibraryA( "AB_ASDx64.dll" );
#else
  DLL_ = LoadLibraryA( "AB_ASD.dll" );
#endif
  if ( !DLL_ )
  {
    throw std::runtime_error( "LoadLibrary failed" );
  }

  mCreateASDLoader = (tCreateASDLoader)GetProcAddress( DLL_, "CreateASDLoader" );
  if ( !mCreateASDLoader )
  {
    throw std::runtime_error( "GetProcAddress failed for CreateASDLoader" );
  }

  mDeleteASDLoader = (tDeleteASDLoader)GetProcAddress( DLL_, "DeleteASDLoader" );
  if ( !mDeleteASDLoader )
  {
    throw std::runtime_error( "GetProcAddress failed for DeleteASDLoader" );
  }

  // Do not throw for the following functions since old libraries won't have them
  mGetASDInterface4 = (tGetASDInterface4)GetProcAddress(DLL_, "GetASDInterface4");
  mGetASDInterface6 = (tGetASDInterface6)GetProcAddress(DLL_, "GetASDInterface4");
}

CASDWrapper::~CASDWrapper()
{
  delete ASDWrapperInterface6_;
  delete ASDWrapperInterface4_;

  std::list<CASDWrapperLoader*>::iterator vLoaderIt = ASDWrapperLoaders_.begin();
  while ( vLoaderIt != ASDWrapperLoaders_.end() )
  {
    delete *vLoaderIt;
    vLoaderIt++;
  }

  FreeLibrary( DLL_ );
}

bool CASDWrapper::CreateASDLoader( const char *Port, TASDType ASDType, IASDLoader **ASDLoader )
{
  bool vRet = mCreateASDLoader( Port, ASDType, ASDLoader );
  if ( vRet )
  {
    CASDWrapperLoader* vLoader = new CASDWrapperLoader( *ASDLoader );
    ASDWrapperLoaders_.push_back( vLoader );
    *ASDLoader = vLoader;
  }
  return vRet;
}

bool CASDWrapper::DeleteASDLoader( IASDLoader *ASDLoader )
{
  std::list<CASDWrapperLoader*>::iterator vLoaderIt = ASDWrapperLoaders_.begin();
  while ( vLoaderIt != ASDWrapperLoaders_.end() )
  {
    if ( *vLoaderIt == ASDLoader )
    {
      bool vDeleteSuccessful = mDeleteASDLoader( ( *vLoaderIt )->GetASDLoader() );
      if ( vDeleteSuccessful )
      {
        delete *vLoaderIt;
        ASDWrapperLoaders_.erase( vLoaderIt );
      }
      return vDeleteSuccessful;
    }
    vLoaderIt++;
  }
  return false;
}

IASDInterface4 *CASDWrapper::GetASDInterface4( IASDLoader *ASDLoader )
{
  CASDWrapperLoader *ASDWrapperLoader = dynamic_cast<CASDWrapperLoader*>(ASDLoader);
  if (!ASDWrapperInterface4_ && mGetASDInterface4 && ASDWrapperLoader)
  {
    ASDWrapperInterface4_ = new CASDWrapperInterface4( mGetASDInterface4(ASDWrapperLoader->GetASDLoader()) );
  }
  return ASDWrapperInterface4_;
}

IASDInterface6 *CASDWrapper::GetASDInterface6( IASDLoader *ASDLoader )
{
  CASDWrapperLoader* ASDWrapperLoader = dynamic_cast<CASDWrapperLoader*>(ASDLoader);
  if (!ASDWrapperInterface6_ && mGetASDInterface6 && ASDWrapperLoader)
  {
    ASDWrapperInterface6_ = new CASDWrapperInterface6(mGetASDInterface6(ASDWrapperLoader->GetASDLoader()));
  }
  return ASDWrapperInterface6_;
}