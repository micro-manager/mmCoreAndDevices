///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperAperture.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERAPERTURE_H_
#define _ASDWRAPPERAPERTURE_H_

#include "ASDInterface.h"

class CASDWrapperFilterSet;

class CASDWrapperAperture : public IApertureInterface
{
public:
  CASDWrapperAperture( IApertureInterface* ApertureInterface );
  ~CASDWrapperAperture();

  // IApertureInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool __stdcall IsSplitFieldAperturePresent();
  IFilterSet* __stdcall GetApertureConfigInterface();

private:
  IApertureInterface* ApertureInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif