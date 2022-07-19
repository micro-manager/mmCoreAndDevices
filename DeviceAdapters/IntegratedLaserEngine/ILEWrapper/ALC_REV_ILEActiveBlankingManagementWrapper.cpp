///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEActiveBlankingManagementWrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_ILEActiveBlankingManagementWrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_ILEActiveBlankingManagementWrapper::CALC_REV_ILEActiveBlankingManagementWrapper( IALC_REV_ILEActiveBlankingManagement* ALC_REV_ILEActiveBlankingManagement ) :
  ALC_REV_ILEActiveBlankingManagement_( ALC_REV_ILEActiveBlankingManagement )
{
  if ( ALC_REV_ILEActiveBlankingManagement_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILEActiveBlankingManagement pointer passed to CALC_REV_ILEActiveBlankingManagementWrapper is null" );
  }
}

CALC_REV_ILEActiveBlankingManagementWrapper::~CALC_REV_ILEActiveBlankingManagementWrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILEActiveBlankingManagement
///////////////////////////////////////////////////////////////////////////////

bool CALC_REV_ILEActiveBlankingManagementWrapper::IsActiveBlankingManagementPresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEActiveBlankingManagement_->IsActiveBlankingManagementPresent( Present );
}

bool CALC_REV_ILEActiveBlankingManagementWrapper::GetNumberOfLines( int *NumberOfLines )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEActiveBlankingManagement_->GetNumberOfLines( NumberOfLines );
}

bool CALC_REV_ILEActiveBlankingManagementWrapper::GetActiveBlankingState( int *EnabledPattern )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEActiveBlankingManagement_->GetActiveBlankingState( EnabledPattern );
}

bool CALC_REV_ILEActiveBlankingManagementWrapper::SetActiveBlankingState( int EnabledPattern )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEActiveBlankingManagement_->SetActiveBlankingState( EnabledPattern );
}
