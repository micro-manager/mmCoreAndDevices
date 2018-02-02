///////////////////////////////////////////////////////////////////////////////
// FILE:          PositionComponentHelper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _POSITIONCOMPONENTHELPER_H_
#define _POSITIONCOMPONENTHELPER_H_

#include <map>
#include <string>

class IFilterSet;


class CPositionComponentHelper
{
public:
  typedef std::string( *tParseDescription )( const std::string& Description );
  typedef std::map<unsigned int, std::string> TPositionNameMap;

  static bool RetrievePositionsFromFilterSet( IFilterSet* FilterSet, TPositionNameMap& PositionNames, tParseDescription ParseDescription );
  static void RetrievePositionsWithoutDescriptions( unsigned int MinValue, unsigned int MaxValue, TPositionNameMap& PositionNames );

private:
  CPositionComponentHelper() {}
  ~CPositionComponentHelper() {}
  static std::string UndefinedPositionBase_;
};

#endif