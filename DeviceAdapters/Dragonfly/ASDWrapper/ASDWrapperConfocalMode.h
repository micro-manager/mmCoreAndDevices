///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperConfocalMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERCONFOCALMODE_H_
#define _ASDWRAPPERCONFOCALMODE_H_

#include "ComponentInterface.h"

class CASDWrapperConfocalMode : public IConfocalModeInterface4
{
public:
  CASDWrapperConfocalMode( IConfocalModeInterface3* ConfocalModeInterface );
  CASDWrapperConfocalMode( IConfocalModeInterface4* ConfocalModeInterface );
  ~CASDWrapperConfocalMode();

  // IConfocalModeInterface
  bool __stdcall ModeNone();
  bool __stdcall ModeConfocalHC();
  bool __stdcall ModeWideField();
  bool __stdcall GetMode( TConfocalMode &BrightFieldMode );

  // IConfocalModeInterface2
  bool __stdcall ModeConfocalHS();
  bool __stdcall IsModeConfocalHSAvailable();
  bool __stdcall IsFirstDisk25um();
  bool __stdcall GetPinHoleSize_um( TConfocalMode ConfocalMode, int *PinHoleSize_um );

  // IConfocalModeInterface3
  bool __stdcall ModeTIRF();
  bool __stdcall IsModeTIRFAvailable();
  bool __stdcall IsConfocalModeAvailable( TConfocalMode Mode );

  // IConfocalModeInterface4
  bool __stdcall ModeBorealisTIRF100();
  bool __stdcall IsModeBorealisTIRF100Available();
  bool __stdcall ModeBorealisTIRF60();
  bool __stdcall IsModeBorealisTIRF60Available();
  bool __stdcall SetConfocalMode( TConfocalMode Mode );

private:
  IConfocalModeInterface3* ConfocalModeInterface3_;
  IConfocalModeInterface4* ConfocalModeInterface4_;
};
#endif