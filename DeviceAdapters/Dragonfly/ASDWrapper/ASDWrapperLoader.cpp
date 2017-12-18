#include "ASDWrapperLoader.h"
#include "ASDWrapperInterface.h"
#include "ASDLoader.h"

CASDWrapperLoader::CASDWrapperLoader( IASDLoader* ASDLoader )
  : ASDLoader_( ASDLoader )
{
  if ( ASDLoader_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDLoader" );
  }
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

IASDLoader* CASDWrapperLoader::GetASDLoader()
{
  return ASDLoader_;
}