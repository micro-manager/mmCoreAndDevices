///////////////////////////////////////////////////////////////////////////////
// FILE:          PortsConfiguration.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _PORTSCONFIGURATION_H_
#define _PORTSCONFIGURATION_H_

#include <string>
#include <map>
#include <vector>

class CPortsConfiguration
{
public:
  typedef std::map<std::string, std::vector<int>> TConfiguration;
  CPortsConfiguration(std::string Name, TConfiguration Configuration);
  ~CPortsConfiguration();

  std::string GetName() const { return Name_; }
  std::vector<std::string> GetPortList() const;
  void GetUnitPortsForMergedPort( std::string MergedPort, std::vector<int>* UnitPorts ) const;
  std::string FindMergedPortForUnitPort( int UnitIndex, int PortIndex ) const;

private:
  std::string Name_;
  TConfiguration Configuration_;
};

std::vector<CPortsConfiguration>& GetPortsConfigurationList();
#endif