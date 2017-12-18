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
  IASDInterface* GetASDInterface();
  IASDInterface2* GetASDInterface2();
  IASDInterface3* GetASDInterface3();
  
private:
  IASDLoader* ASDLoader_;
  CASDWrapperInterface* ASDInterface_;
};

#endif