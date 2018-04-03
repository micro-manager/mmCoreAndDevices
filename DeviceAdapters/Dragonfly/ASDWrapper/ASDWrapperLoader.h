///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperLoader.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERLOADER_H_
#define _ASDWRAPPERLOADER_H_

#include "ASDLoader.h"

class IASDInterface;
class IASDInterface3;
class CASDWrapperInterface;

class CASDWrapperLoader : public IASDLoader
{
public:
  CASDWrapperLoader( IASDLoader* ASDLoader );
  ~CASDWrapperLoader();

  // IASDLoader
  IASDInterface* __stdcall GetASDInterface();
  IASDInterface2* __stdcall GetASDInterface2();
  IASDInterface3* __stdcall GetASDInterface3();

  IASDLoader* GetASDLoader();
  
private:
  IASDLoader* ASDLoader_;
  CASDWrapperInterface* ASDInterface_;
};

#endif