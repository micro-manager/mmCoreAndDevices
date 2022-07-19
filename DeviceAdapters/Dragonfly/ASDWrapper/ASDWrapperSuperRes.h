///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperSuperRes.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERSUPERRES_H_
#define _ASDWRAPPERSUPERRES_H_

#include "ASDInterface.h"


class CASDWrapperSuperRes : public ISuperResInterface
{
public:
  CASDWrapperSuperRes( ISuperResInterface* SuperResInterface );
  ~CASDWrapperSuperRes();

  // ISuperResInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );

private:
  ISuperResInterface* SuperResInterface_;
};
#endif