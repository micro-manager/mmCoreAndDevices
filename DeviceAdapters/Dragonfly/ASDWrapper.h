#ifndef _ASDWRAPPER_H_
#define _ASDWRAPPER_H_

#include "ASDLoader.h"
#include "Windows.h"

class CASDWrapper 
{
public:
  CASDWrapper();
  ~CASDWrapper();

  bool CreateASDLoader( const char *Port, TASDType ASDType, IASDLoader **ASDLoader );
  bool DeleteASDLoader( IASDLoader *ASDLoader );

private:
  HMODULE mDLL;

  typedef bool( __stdcall *tCreateASDLoader )( const char *Port, TASDType ASDType, IASDLoader **ASDLoader );
  tCreateASDLoader mCreateASDLoader;

  typedef bool( __stdcall *tDeleteASDLoader )( IASDLoader *ASDLoader );
  tDeleteASDLoader mDeleteASDLoader;
};
#endif