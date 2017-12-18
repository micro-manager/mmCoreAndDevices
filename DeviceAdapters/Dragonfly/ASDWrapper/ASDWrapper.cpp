#include "ASDWrapper.h"
#include <stdexcept>

CASDWrapper::CASDWrapper()
  : mDLL( nullptr ),
  mCreateASDLoader( nullptr )
{
#ifdef _M_X64
  HMODULE mDLL = LoadLibraryA( "AB_ASDx64.dll" );
#else
  HMODULE mDLL = LoadLibraryA( "AB_ASD.dll" );
#endif
  if ( !mDLL )
  {
    throw std::runtime_error( "LoadLibrary failed" );
  }

  mCreateASDLoader = (tCreateASDLoader)GetProcAddress( mDLL, "CreateASDLoader" );
  if ( !mCreateASDLoader )
  {
    throw std::runtime_error( "GetProcAddress failed for CreateASDLoader" );
  }

  mDeleteASDLoader = (tDeleteASDLoader)GetProcAddress( mDLL, "DeleteASDLoader" );
  if ( !mDeleteASDLoader )
  {
    throw std::runtime_error( "GetProcAddress failed for DeleteASDLoader" );
  }
}

CASDWrapper::~CASDWrapper()
{
  FreeLibrary( mDLL );
}

bool CASDWrapper::CreateASDLoader( const char *Port, TASDType ASDType, IASDLoader **ASDLoader )
{
  return mCreateASDLoader( Port, ASDType, ASDLoader );
}

bool CASDWrapper::DeleteASDLoader( IASDLoader *ASDLoader )
{
  return mDeleteASDLoader( ASDLoader );
}