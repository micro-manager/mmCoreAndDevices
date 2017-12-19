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
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool IsSplitFieldAperturePresent();
  IFilterSet* GetApertureConfigInterface();

private:
  IApertureInterface* ApertureInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif