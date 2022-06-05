///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperConfocalMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERCONFOCALMODE_H_
#define _ASDWRAPPERCONFOCALMODE_H_

#include "ComponentInterface.h"

class CASDWrapperConfocalMode : public IConfocalModeInterface3
{
public:
  CASDWrapperConfocalMode( IConfocalModeInterface3* ConfocalModeInterface );
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

private:
  IConfocalModeInterface3* ConfocalModeInterface_;
};

class CASDWrapperConfocalMode4 : public IConfocalModeInterface4
{
public:
  CASDWrapperConfocalMode4(IConfocalModeInterface4* ConfocalModeInterface);
  ~CASDWrapperConfocalMode4();

  // IConfocalModeInterface
  bool __stdcall ModeNone() { return ASDWrapperConfocalMode_.ModeNone(); }
  bool __stdcall ModeConfocalHC() { return ASDWrapperConfocalMode_.ModeConfocalHC(); }
  bool __stdcall ModeWideField() { return ASDWrapperConfocalMode_.ModeWideField(); }
  bool __stdcall GetMode(TConfocalMode& BrightFieldMode) { return ASDWrapperConfocalMode_.GetMode(BrightFieldMode); }

  // IConfocalModeInterface2
  bool __stdcall ModeConfocalHS() { return ASDWrapperConfocalMode_.ModeConfocalHS(); }
  bool __stdcall IsModeConfocalHSAvailable() { return ASDWrapperConfocalMode_.IsModeConfocalHSAvailable(); }
  bool __stdcall IsFirstDisk25um() { return ASDWrapperConfocalMode_.IsFirstDisk25um(); }
  bool __stdcall GetPinHoleSize_um(TConfocalMode ConfocalMode, int* PinHoleSize_um) { return ASDWrapperConfocalMode_.GetPinHoleSize_um(ConfocalMode, PinHoleSize_um); }

  // IConfocalModeInterface3
  bool __stdcall ModeTIRF() { return ASDWrapperConfocalMode_.ModeTIRF(); }
  bool __stdcall IsModeTIRFAvailable() { return ASDWrapperConfocalMode_.IsModeTIRFAvailable(); }
  bool __stdcall IsConfocalModeAvailable(TConfocalMode Mode) { return ASDWrapperConfocalMode_.IsConfocalModeAvailable(Mode); }

  // IConfocalModeInterface4
  bool __stdcall ModeBorealisTIRF100();
  bool __stdcall IsModeBorealisTIRF100Available();
  bool __stdcall ModeBorealisTIRF60();
  bool __stdcall IsModeBorealisTIRF60Available();
  bool __stdcall SetConfocalMode(TConfocalMode Mode);

private:
  CASDWrapperConfocalMode ASDWrapperConfocalMode_;
  IConfocalModeInterface4* ConfocalModeInterface4_;
};
#endif