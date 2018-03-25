///////////////////////////////////////////////////////////////////////////////
// FILE:          PortsConfiguration.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "PortsConfiguration.h"
#include "IntegratedLaserEngine.h"
#include <windows.h>
#include <Shlobj.h>
#include <fstream>


#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 

CPortsConfiguration::CPortsConfiguration( const std::string& Device1Name, const std::string& Device2Name, CIntegratedLaserEngine* MMILE ) :
  Device1Name_( Device1Name ),
  Device2Name_( Device2Name ),
  MMILE_( MMILE )
{
  std::string vBaseFileName = "DualILE - ";
  std::string vFile1Name = vBaseFileName + Device1Name_ + ", " + Device2Name_ + ".desc";
  std::string vFile2Name = vBaseFileName + Device2Name_ + ", " + Device1Name_ + ".desc";
  PWSTR vWPathName;
  if ( SUCCEEDED( SHGetKnownFolderPath( FOLDERID_PublicDocuments, KF_FLAG_DEFAULT, NULL, &vWPathName ) ) ) // FOLDERID_PublicDocuments for Public documents path - FOLDERID_ProgramData for All users (aka ProgramData) path
  {
    std::wstring vWPathNameString = vWPathName;
    vBaseFileName.assign( vWPathNameString.begin(), vWPathNameString.end() );
    CoTaskMemFree( vWPathName );
    std::string vFullPath = vBaseFileName + PATH_SEPARATOR + "Fusion Global Data" + PATH_SEPARATOR + "Config" + PATH_SEPARATOR;
    std::string vFileName = vFullPath + vFile1Name;
    if ( !LoadConfigFile( vFileName ) )
    {
      vFileName = vFullPath + vFile2Name;
      if ( !LoadConfigFile( vFileName ) )
      {
        // Look for the config file in the MicroManager folder
        if ( !LoadConfigFile( vFile1Name ) )
        {
          if ( !LoadConfigFile( vFile2Name ) )
          {
            // No config file found
            MMILE_->LogMMMessage( "No config file found" );
            //TODO: Handle this case when we know what we want to do
          }
        }
      }
    }
  }


}

CPortsConfiguration::~CPortsConfiguration()
{
}

bool CPortsConfiguration::LoadConfigFile(const std::string& FileName)
{
  static const std::string vKeyBaseName = "OutputPort";
  TConfiguration vNewConfiguration;

  std::ifstream vFile( FileName );
  if ( !vFile.is_open() )
  {
    MMILE_->LogMMMessage( "Failed to open port config file: " + FileName );
    return false;
  }

  std::string vLine;
  while ( std::getline( vFile, vLine ) )
  {
    size_t vEqualPos = vLine.find( "=" );
    if ( vEqualPos != std::string::npos )
    {
      if ( vLine.find( vKeyBaseName ) != std::string::npos )
      {
        bool vPortConfigurationRetrieved = false;
        std::vector<std::string> vValueList;
        std::string vValueString = vLine.substr( vEqualPos + 1 );
        std::istringstream vIss( vValueString );
        for ( std::string vToken; std::getline( vIss, vToken, ',' ); )
        {
          vValueList.push_back( vToken );
        }
        if ( vValueList.size() >= 3 )
        {
          try
          {
            vNewConfiguration[vValueList[0]] = std::vector<int>();
            vNewConfiguration[vValueList[0]].push_back( std::atoi( vValueList[1].c_str() ) );
            vNewConfiguration[vValueList[0]].push_back( std::atoi( vValueList[2].c_str() ) );
            MMILE_->LogMMMessage( "Loaded port from config file: Port " + vValueList[0] + ": unit1/" + vValueList[1] + " - unit2/" + vValueList[2], true );
            vPortConfigurationRetrieved = true;
          }
          catch ( ... )
          {}
        }
        if ( !vPortConfigurationRetrieved )
        {
          MMILE_->LogMMMessage( "Corrupted port configuration entry. Entry: " + vValueString );
        }
      }
    }
  }

  if ( !vNewConfiguration.empty() )
  {
    Configuration_ = vNewConfiguration;
    return true;
  }
  return false;
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

void CPortsConfiguration::GetUnitPortsForMergedPort( const std::string& MergedPort, std::vector<int>* UnitPorts ) const
{
  TConfiguration::const_iterator vMergedPortIt = Configuration_.find( MergedPort );
  if ( vMergedPortIt != Configuration_.end() )
  {
    *UnitPorts = vMergedPortIt->second;
  }
}

std::string CPortsConfiguration::FindMergedPortForUnitPort( int Unit1Port, int Unit2Port ) const
{
  std::string vMergedPort = "";
  bool vPortFound = false;
  TConfiguration::const_iterator vConfigurationIt = Configuration_.begin();
  while ( !vPortFound && vConfigurationIt != Configuration_.end() )
  {
    if ( vConfigurationIt->second.size() >= 2 )
    {
      if ( vConfigurationIt->second[0] == Unit1Port && vConfigurationIt->second[1] == Unit2Port )
      {
        vMergedPort = vConfigurationIt->first;
        vPortFound = true;
      }
    }
    ++vConfigurationIt;
  }
  return vMergedPort;
}