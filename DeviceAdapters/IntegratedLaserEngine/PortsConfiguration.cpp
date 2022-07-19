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
#include "boost\filesystem.hpp"


#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 

const std::string CPortsConfiguration::ConfigFileKeyBaseName_ = "OutputPort";


CPortsConfiguration::CPortsConfiguration( const std::string& Device1Name, const std::string& Device2Name, CIntegratedLaserEngine* MMILE ) :
  Device1Name_( Device1Name ),
  Device2Name_( Device2Name ),
  MMILE_( MMILE )
{  
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CPortsConfiguration: Pointer tomain class invalid" );
  }

  bool vGenerateDefaultConfig = false;
  std::string vFile1Name = GenerateFileName( Device1Name_, Device2Name_ );
  std::string vFile2Name = GenerateFileName( Device2Name_, Device1Name_ );
  // Look for the config file in the MicroManager folder
  if ( !LoadConfigFile( vFile1Name ) )
  {
    if ( !LoadConfigFile( vFile2Name ) )
    {
      // Look for the config file in the Fusion config folder
      std::string vFusionConfigFolderPath;
      if ( GetFusionConfigFolderPath( &vFusionConfigFolderPath ) )
      {
        std::string vFileName = vFusionConfigFolderPath + vFile1Name;
        if ( !LoadConfigFile( vFileName ) )
        {
          vFileName = vFusionConfigFolderPath + vFile2Name;
          if ( !LoadConfigFile( vFileName ) )
          {
            // No config file found, generate a demo file
            vGenerateDefaultConfig = true;
            MMILE_->LogMMMessage( "No port configuration file found. Using a default port configuration." );
          }
        }
      }
      else
      {
        vGenerateDefaultConfig = true;
      }
    }
  }
  if ( vGenerateDefaultConfig )
  {
    GenerateDefaultConfig();
  }
}

CPortsConfiguration::~CPortsConfiguration()
{
}

std::string CPortsConfiguration::GenerateFileName( const std::string& Device1Name, const std::string& Device2Name ) const
{
  static const std::string vBaseFileName = "DualILE - ";
  return vBaseFileName + Device1Name + ", " + Device2Name + ".desc";
}

bool CPortsConfiguration::GetFusionConfigFolderPath( std::string* Path ) const
{
  PWSTR vWPathName;
  if ( SUCCEEDED( SHGetKnownFolderPath( FOLDERID_PublicDocuments, KF_FLAG_DEFAULT, NULL, &vWPathName ) ) ) // FOLDERID_PublicDocuments for Public documents path - FOLDERID_ProgramData for All users (aka ProgramData) path
  {
    static const size_t vMaxBufferByteSize = 256;
    char* vPathName = new char[vMaxBufferByteSize];
    size_t vResultStringByteSize;
    errno_t vError = wcstombs_s( &vResultStringByteSize, vPathName, vMaxBufferByteSize, vWPathName, vMaxBufferByteSize - 1 );
    if ( vError != 0 )
    {
      MMILE_->LogMMMessage( "Failed to convert Fusion config file path. Conversion result: " + std::string( vPathName ), true );
      return false;
    }
    CoTaskMemFree( vWPathName );
    *Path = std::string( vPathName ) + PATH_SEPARATOR + "Fusion Global Data" + PATH_SEPARATOR + "Config" + PATH_SEPARATOR;
    return true;
  }
  return false;
}

bool CPortsConfiguration::TestFileWriteAccess( const std::string& FileName, std::string* FileToOpen ) const
{
  std::ofstream vFile( FileName );
  if ( vFile.is_open() )
  {
    vFile.close();
    *FileToOpen = FileName;
    return true;
  }

  // Create Fusion config folder if needed and test if we can write the config file 
  //std::string vPath;
  //if ( GetFusionConfigFolderPath( &vPath ) )
  //{
  //  std::wstring vwPath( vPath.begin(), vPath.end() );
  //  boost::filesystem::path vDirectory( vwPath.c_str() );
  //  if ( !boost::filesystem::exists( vDirectory ) )
  //  {
  //    if ( !boost::filesystem::create_directory( vDirectory ) )
  //    {
  //      return false;
  //    }
  //  }
  //  std::string vFullFilePath = vPath + FileName;
  //  std::ofstream vFile( vFullFilePath );
  //  if ( vFile.is_open() )
  //  {
  //    vFile.close();
  //    *FileToOpen = vFullFilePath;
  //    return true;
  //  }
  //}
  return false;
}

void CPortsConfiguration::GenerateDefaultConfig()
{
  std::vector<int> vTIRFConfiguration; vTIRFConfiguration.push_back( 1 ); vTIRFConfiguration.push_back( 1 );
  std::vector<int> vBorealisConfiguration; vBorealisConfiguration.push_back( 3 ); vBorealisConfiguration.push_back( 1 );
  std::vector<int> vPhotoStimulationConfiguration; vPhotoStimulationConfiguration.push_back( 2 ); vPhotoStimulationConfiguration.push_back( 2 );
  Configuration_["TIRF"] = vTIRFConfiguration;
  Configuration_["Borealis"] = vBorealisConfiguration;
  Configuration_["Photostimulation"] = vPhotoStimulationConfiguration;

  std::string vFileName = GenerateFileName( Device1Name_, Device2Name_ );
  std::string vFileToOpen;
  if ( TestFileWriteAccess( vFileName, &vFileToOpen ) )
  {
    std::ofstream vFile( vFileToOpen );
    WritePortToConfigFile( vFile, 0, "TIRF" );
    WritePortToConfigFile( vFile, 1, "Borealis" );
    WritePortToConfigFile( vFile, 2, "Photostimulation" );    
    vFile.close();
  }
  else
  {
    MMILE_->LogMMMessage( "Generating configuration file FAILED" );
  }
}

void CPortsConfiguration::WritePortToConfigFile( std::ofstream& File, int PortIndex, const std::string& PortName ) const
{
  if ( File.is_open() )
  {
    TConfiguration::const_iterator vPortIt = Configuration_.find( PortName );
    if ( vPortIt != Configuration_.end() && vPortIt->second.size() >= 2)
    {
      File << ConfigFileKeyBaseName_ << "[" + std::to_string( static_cast<long long>( PortIndex ) ) + "]=" << vPortIt->first << "," << vPortIt->second[0] << "," << vPortIt->second[1] << std::endl;
    }
  }
}

bool CPortsConfiguration::LoadConfigFile(const std::string& FileName)
{
  TConfiguration vNewConfiguration;

  std::ifstream vFile( FileName );
  if ( !vFile.is_open() )
  {
    MMILE_->LogMMMessage( "Failed to open port config file: " + FileName, true );
    return false;
  }

  std::string vLine;
  while ( std::getline( vFile, vLine ) )
  {
    size_t vEqualPos = vLine.find( "=" );
    if ( vEqualPos != std::string::npos )
    {
      if ( vLine.find( ConfigFileKeyBaseName_ ) != std::string::npos )
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