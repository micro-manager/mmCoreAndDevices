///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperLoader.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERLOADER_H_
#define _ASDWRAPPERLOADER_H_

#include "ASDLoader.h"
#include "Windows.h"

class IASDInterface;
class IASDInterface3;
class CASDWrapperInterface;

class CASDWrapperLoader : public IASDLoader
{
public:
  CASDWrapperLoader( IASDLoader* ASDLoader, HMODULE DLL );
  ~CASDWrapperLoader();

  // IASDLoader
  IASDInterface* __stdcall GetASDInterface();
  IASDInterface2* __stdcall GetASDInterface2();
  IASDInterface3* __stdcall GetASDInterface3();
  IASDInterface4* __stdcall GetASDInterface4();
  IASDInterface6* __stdcall GetASDInterface6();

  IASDLoader* GetASDLoader();

  typedef IASDInterface4* ( __stdcall* tGetASDInterface4 )( IASDLoader* ASDLoader );
  tGetASDInterface4 mGetASDInterface4;

  typedef IASDInterface6* ( __stdcall* tGetASDInterface6 )( IASDLoader* ASDLoader );
  tGetASDInterface6 mGetASDInterface6;

private:
  IASDLoader* ASDLoader_;
  CASDWrapperInterface* ASDInterface_;
};

#endif