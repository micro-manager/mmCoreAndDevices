#include "ASDWrapperLoader.h"
#include "ASDWrapperInterface.h"
#include "ASDLoader.h"

CASDWrapperLoader::CASDWrapperLoader( IASDLoader* ASDLoader, HMODULE DLL )
  : ASDLoader_( ASDLoader )
{
  if ( ASDLoader_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDLoader" );
  }

  // Do not throw for the following functions since old libraries won't have them
  mGetASDInterface4 = ( tGetASDInterface4 ) GetProcAddress( DLL, "GetASDInterface4" );
  mGetASDInterface6 = ( tGetASDInterface6 ) GetProcAddress( DLL, "GetASDInterface6" );

  if ( mGetASDInterface6 && mGetASDInterface6( ASDLoader_ ) != nullptr )
    ASDInterface_ = new CASDWrapperInterface( mGetASDInterface6( ASDLoader_ ) );
  else if ( mGetASDInterface4 && mGetASDInterface4( ASDLoader_ ) != nullptr )
    ASDInterface_ = new CASDWrapperInterface( mGetASDInterface4( ASDLoader_ ) );
  else
    ASDInterface_ = new CASDWrapperInterface( ASDLoader_->GetASDInterface3() );
}

CASDWrapperLoader::~CASDWrapperLoader()
{
  delete ASDInterface_;
}

IASDInterface* CASDWrapperLoader::GetASDInterface()
{
  return ASDInterface_;
}

IASDInterface2* CASDWrapperLoader::GetASDInterface2()
{
  return ASDInterface_;
}

IASDInterface3* CASDWrapperLoader::GetASDInterface3()
{
  return ASDInterface_;
}

IASDInterface4* CASDWrapperLoader::GetASDInterface4()
{
  return ASDInterface_->IsASDInterface4Available() ? ASDInterface_ : nullptr;
}

IASDInterface6* CASDWrapperLoader::GetASDInterface6()
{
  return ASDInterface_->IsASDInterface6Available() ? ASDInterface_ : nullptr;
}

IASDLoader* CASDWrapperLoader::GetASDLoader()
{
  return ASDLoader_;
}