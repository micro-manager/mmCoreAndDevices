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
  typedef std::map<unsigned int, std::string> TPositionNameMap;

  static bool RetrievePositionsFromFilterSet( IFilterSet* FilterSet, TPositionNameMap& PositionNames, bool AddIndexToPositionNames );
  static void RetrievePositionsWithoutDescriptions( unsigned int MinValue, unsigned int MaxValue, TPositionNameMap& PositionNames );

private:
  CPositionComponentHelper() {}
  ~CPositionComponentHelper() {}
  static std::string UndefinedPositionBase_;
};

#endif