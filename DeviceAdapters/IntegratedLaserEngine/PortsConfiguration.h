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

class CIntegratedLaserEngine;

class CPortsConfiguration
{
public:
  typedef std::map<std::string, std::vector<int>> TConfiguration;
  CPortsConfiguration( const std::string& Device1Name, const std::string& Device2Name, CIntegratedLaserEngine* MMILE );
  ~CPortsConfiguration();

  std::vector<std::string> GetPortList() const;
  void GetUnitPortsForMergedPort( const std::string& MergedPort, std::vector<int>* UnitPorts ) const;
  std::string FindMergedPortForUnitPort( int Uni1Port, int Unit2Port ) const;

  bool LoadConfigFile( const std::string& FileName );

private:
  std::string Device1Name_;
  std::string Device2Name_;
  CIntegratedLaserEngine* MMILE_;
  TConfiguration Configuration_;
  static const std::string ConfigFileKeyBaseName_;

  std::string GenerateFileName( const std::string& Device1Name, const std::string& Device2Name ) const;
  bool GetFusionConfigFolderPath( std::string* Path ) const;
  bool TestFileWriteAccess( const std::string& FileName, std::string* FileToOpen = nullptr ) const;
  void GenerateDefaultConfig();
  void WritePortToConfigFile( std::ofstream& File, int PortIndex, const std::string& PortName ) const;
};

#endif