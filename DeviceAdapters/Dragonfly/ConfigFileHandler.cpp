#include "ConfigFileHandler.h"
#include <fstream>
#include <windows.h>
#include "Dragonfly.h"
#include <Shlobj.h>

using namespace std;

const char* const g_SectionName = "config";
const char* const g_DefaultConfigFileName = "DragonflyMMConfig.ini";
const char* const g_ConfigFile = "Configuration file";

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 

CConfigFileHandler::CConfigFileHandler( CDragonfly* MMDragonfly )
  : MMDragonfly_( MMDragonfly )
{
  FileName_ = "Undefined";

  // Create property with a default path set to My Documents
  PWSTR vWPathName;
  if ( SUCCEEDED( SHGetKnownFolderPath( FOLDERID_ProgramData, KF_FLAG_DEFAULT, NULL, &vWPathName ) ) ) // FOLDERID_PublicDocuments for Public documents path - FOLDERID_ProgramData for All users (aka ProgramData) path
  {
    static const size_t vMaxBufferByteSize = 256;
    char* vPathName = new char[vMaxBufferByteSize];
    size_t vResultStringByteSize;
    errno_t vError = wcstombs_s( &vResultStringByteSize, vPathName, vMaxBufferByteSize, vWPathName, vMaxBufferByteSize - 1 );
    if ( vError != 0 )
    {
      throw std::runtime_error( "String conversion error during Configuration File path property creation" );
    }
    FileName_ = string( vPathName ) + PATH_SEPARATOR + string( g_DefaultConfigFileName );
    CoTaskMemFree( vWPathName );
  }
  CPropertyAction* vAct = new CPropertyAction( this, &CConfigFileHandler::OnConfigFileChange );
  int vRet = MMDragonfly_->CreateProperty( g_ConfigFile, FileName_.c_str(), MM::String, false, vAct, true );
  if ( vRet != DEVICE_OK )
  {
    throw std::runtime_error( "Error creating Configuration File path property" );
  }
}

CConfigFileHandler::~CConfigFileHandler()
{
  Config_.clear();
}

int CConfigFileHandler::OnConfigFileChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( FileName_.c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( FileName_ );
  }
  return vRet;
}

bool CConfigFileHandler::LoadPropertyValue( const std::string& PropertyName, std::string& PropertyValue )
{
  if ( Config_.find( PropertyName ) != Config_.end() )
  {
    PropertyValue = Config_[PropertyName];
    return true;
  }
  return false;
}

void CConfigFileHandler::SavePropertyValue( const std::string& PropertyName, const std::string& PropertyValue )
{
  Config_[PropertyName] = PropertyValue;
  SaveConfig();
}

int CConfigFileHandler::TestFileAccess( const std::string& FileName ) const
{
  // Try to write a test section to the file
  if ( !WritePrivateProfileSection( "test", "test=test", FileName.c_str() ) )
  {
    return ERR_CONFIGFILEIO_ERROR;
  }
  // Erase the test section from the file
  WritePrivateProfileSection( "test", NULL, FileName.c_str() );
  return DEVICE_OK;
}

int CConfigFileHandler::LoadConfig()
{
  int vRet = TestFileAccess( FileName_ );
  if ( vRet == DEVICE_OK )
  {
    Config_.clear();
    // Load configuration from file
    const int vBufferSize = 10000;
    char vBuffer[vBufferSize] = "";
    int vCharsRead = 0;
    vCharsRead = GetPrivateProfileSection( g_SectionName, vBuffer, vBufferSize, FileName_.c_str() );
    if ( ( vCharsRead > 0 ) && ( vCharsRead < ( vBufferSize - 2 ) ) )
    {
      // Buffer contains string of format: key1=value1\0key2=value2
      const char* vSubstr = vBuffer;
      while ( *vSubstr != '\0' )
      {
        size_t vSubstrLen = strlen( vSubstr );
        const char* vValuePos = strchr( vSubstr, '=' );
        if ( vValuePos != nullptr )
        {
          char vKey[256] = "";
          char vValue[256] = "";
          strncpy( vKey, vSubstr, vValuePos - vSubstr );
          strncpy( vValue, vValuePos + 1, vSubstrLen - ( vValuePos - vSubstr ) );
          Config_[vKey] = vValue;
        }
        vSubstr += ( vSubstrLen + 1 );
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Dragonfly Config ini file empty, corrupted or not found" );
    }
  }
  return vRet;
}

void CConfigFileHandler::SaveConfig()
{
  map<string, string>::const_iterator vConfigIt = Config_.begin();
  while ( vConfigIt != Config_.end() )
  {
    if ( !WritePrivateProfileString( g_SectionName, vConfigIt->first.c_str(), vConfigIt->second.c_str(), FileName_.c_str() ) )
    {
      MMDragonfly_->LogComponentMessage( "Failed to write data to Dragonfly ini file. Error returned: " + to_string( static_cast< long long >( GetLastError() ) ) );
    }
    ++vConfigIt;
  }
}
