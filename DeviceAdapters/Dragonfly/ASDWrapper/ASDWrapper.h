///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPER_H_
#define _ASDWRAPPER_H_

#include "ASDLoader.h"
#include "Windows.h"
#include <list>

class CASDWrapperLoader;
class CASDWrapperInterface4;
class CASDWrapperInterface6;

class CASDWrapper 
{
public:
  CASDWrapper();
  ~CASDWrapper();

  bool CreateASDLoader( const char *Port, TASDType ASDType, IASDLoader **ASDLoader );
  bool DeleteASDLoader( IASDLoader *ASDLoader );
  IASDInterface4 *GetASDInterface4(IASDLoader *ASDLoader);
  IASDInterface6 *GetASDInterface6(IASDLoader *ASDLoader);

private:
  HMODULE DLL_;
  std::list<CASDWrapperLoader*> ASDWrapperLoaders_;
  CASDWrapperInterface4* ASDWrapperInterface4_;
  CASDWrapperInterface6* ASDWrapperInterface6_;

  typedef bool( __stdcall *tCreateASDLoader )( const char *Port, TASDType ASDType, IASDLoader **ASDLoader );
  tCreateASDLoader mCreateASDLoader;

  typedef bool( __stdcall *tDeleteASDLoader )( IASDLoader *ASDLoader );
  tDeleteASDLoader mDeleteASDLoader;

  typedef IASDInterface4*( __stdcall *tGetASDInterface4 )( IASDLoader *ASDLoader );
  tGetASDInterface4 mGetASDInterface4;

  typedef IASDInterface6*( __stdcall *tGetASDInterface6 )( IASDLoader *ASDLoader );
  tGetASDInterface6 mGetASDInterface6;
};

#endif