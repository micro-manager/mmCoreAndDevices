///////////////////////////////////////////////////////////////////////////////
// FILE:          ConfigFileHandlerInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _CONFIGFILEHANDLERINTERFACE_H_
#define _CONFIGFILEHANDLERINTERFACE_H_

#include <string>

class IConfigFileHandler
{
public:
  virtual bool LoadPropertyValue( const std::string& PropertyName, std::string& PropertyValue ) = 0;
  virtual void SavePropertyValue( const std::string& PropertyName, const std::string& PropertyValue ) = 0;
};
#endif