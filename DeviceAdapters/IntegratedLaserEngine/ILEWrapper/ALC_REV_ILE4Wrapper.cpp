///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILE4Wrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifdef WIN32
#include <windows.h>
#endif
#include "ALC_REV_ILE4Wrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_ILE4Wrapper::CALC_REV_ILE4Wrapper( IALC_REV_ILE4* ALC_REV_ILE4 ) :
  ALC_REV_ILE4_( ALC_REV_ILE4 )
{
  if ( ALC_REV_ILE4_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILE4 pointer passed to CALC_REV_ILE4Wrapper is null" );
  }
}

CALC_REV_ILE4Wrapper::~CALC_REV_ILE4Wrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILE4
///////////////////////////////////////////////////////////////////////////////

bool CALC_REV_ILE4Wrapper::GetNumberOfUnits( int *NumUnits )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE4_->GetNumberOfUnits( NumUnits );
}

bool CALC_REV_ILE4Wrapper::IsActiveBlankingManagementPresent( int UnitIndex, bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE4_->IsActiveBlankingManagementPresent( UnitIndex, Present );
}

bool CALC_REV_ILE4Wrapper::GetNumberOfLines( int UnitIndex, int *NumberOfLines )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE4_->GetNumberOfLines( UnitIndex, NumberOfLines );
}

bool CALC_REV_ILE4Wrapper::GetActiveBlankingState( int UnitIndex, int *EnabledPattern )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE4_->GetActiveBlankingState( UnitIndex, EnabledPattern );
}

bool CALC_REV_ILE4Wrapper::SetActiveBlankingState( int UnitIndex, int EnabledPattern )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE4_->SetActiveBlankingState( UnitIndex, EnabledPattern );
}
