#include "ASDWrapper.h"
#include "ASDWrapperLoader.h"
#include <stdexcept>

CASDWrapper::CASDWrapper()
  : DLL_( nullptr ),
  mCreateASDLoader( nullptr ),
  mDeleteASDLoader( nullptr )
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
}

CASDWrapper::~CASDWrapper()
{
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