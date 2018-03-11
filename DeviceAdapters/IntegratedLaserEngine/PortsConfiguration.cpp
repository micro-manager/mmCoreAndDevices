///////////////////////////////////////////////////////////////////////////////
// FILE:          PortsConfiguration.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "PortsConfiguration.h"


// Default configurations
CPortsConfiguration DraongonflyConfiguration( "Dragonfly", { { "Dragonfly",{ 1, 1 } },{ "Microscope Side",{ 0, 2 } },{ "Microscope Rear",{ 2, 0 } },{ "TIRF",{ 3, 0 } },{ "Mosaic",{ 0, 3 } } } );
CPortsConfiguration DUALTESTConfiguration( "DUALTEST", { { "TEST_12",{ 1, 2 } },{ "TEST_23",{ 2, 3 } },{ "TEST_31",{ 3, 1 } } } );
CPortsConfiguration SINGLETESTConfiguration( "SINGLETEST", { { "TEST_10",{ 1, 0 } },{ "TEST_20",{ 2, 0 } },{ "TEST_30",{ 3, 0 } },{ "TEST_01",{ 0, 1 } },{ "TEST_02",{ 0, 2 } },{ "TEST_03",{ 0, 3 } } } );

std::vector<CPortsConfiguration> PortsConfigurationList = { DraongonflyConfiguration, DUALTESTConfiguration, SINGLETESTConfiguration };

std::vector<CPortsConfiguration>& GetPortsConfigurationList()
{
  return PortsConfigurationList;
}

CPortsConfiguration::CPortsConfiguration( std::string Name, TConfiguration Configuration ) :
  Name_( Name ),
  Configuration_( Configuration )
{
}

CPortsConfiguration::~CPortsConfiguration()
{
}

std::vector<std::string> CPortsConfiguration::GetPortList() const
{
  std::vector<std::string> vPortList;
  TConfiguration::const_iterator vConfigurationIt = Configuration_.begin();
  while ( vConfigurationIt != Configuration_.end() )
  {
    vPortList.push_back(vConfigurationIt->first);
    ++vConfigurationIt;
  }
  return vPortList;
}

void CPortsConfiguration::GetUnitPortsForMergedPort( std::string MergedPort, std::vector<int>* UnitPorts ) const
{
  TConfiguration::const_iterator vMergedPortIt = Configuration_.find( MergedPort );
  if ( vMergedPortIt != Configuration_.end() )
  {
    *UnitPorts = vMergedPortIt->second;
  }
}

std::string CPortsConfiguration::FindMergedPortForUnitPort( int UnitIndex, int PortIndex ) const
{
  std::string vMergedPort = "";
  bool vPortFound = false;
  TConfiguration::const_iterator vConfigurationIt = Configuration_.begin();
  while ( !vPortFound && vConfigurationIt != Configuration_.end() )
  {
    if ( UnitIndex < vConfigurationIt->second.size() )
    {
      if ( vConfigurationIt->second[UnitIndex] == PortIndex )
      {
        vMergedPort = vConfigurationIt->first;
        vPortFound = true;
      }
    }
    ++vConfigurationIt;
  }
  return vMergedPort;
}