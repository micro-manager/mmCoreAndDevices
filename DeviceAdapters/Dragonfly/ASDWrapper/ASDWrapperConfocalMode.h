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
  bool ModeNone();
  bool ModeConfocalHC();
  bool ModeWideField();
  bool GetMode( TConfocalMode &BrightFieldMode );

  // IConfocalModeInterface2
  bool ModeConfocalHS();
  bool IsModeConfocalHSAvailable();
  bool IsFirstDisk25um();
  bool GetPinHoleSize_um( TConfocalMode ConfocalMode, int *PinHoleSize_um );

  // IConfocalModeInterface3
  bool ModeTIRF();
  bool IsModeTIRFAvailable();
  bool IsConfocalModeAvailable( TConfocalMode Mode );

private:
  IConfocalModeInterface3* ConfocalModeInterface_;
};
#endif