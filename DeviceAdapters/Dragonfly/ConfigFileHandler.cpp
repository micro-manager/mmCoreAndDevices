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

  //TCHAR vPathName[MAX_PATH];
  //if ( GetTempPath( MAX_PATH, vPathName ) != 0 )
  WCHAR vWPathName[MAX_PATH];
  if ( SUCCEEDED( SHGetFolderPathW( NULL, CSIDL_PERSONAL, NULL, 0, vWPathName ) ) ) // CSIDL_PROFILE for users path - CSIDL_PERSONAL for my documents
  {
    wstring vWPathNameString = vWPathName;
    FileName_.assign( vWPathNameString.begin(), vWPathNameString.end() );
    FileName_ = FileName_ + PATH_SEPARATOR + string( g_DefaultConfigFileName );
  }
  CPropertyAction* vAct = new CPropertyAction( this, &CConfigFileHandler::OnConfigFileChange );
  MMDragonfly_->CreateProperty( g_ConfigFile, FileName_.c_str(), MM::String, false, vAct, true );
  //RequestedFileName_ = FileName_;
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
    MMDragonfly_->LogComponentMessage( "Before Get: " + FileName_ );
  }
  else if ( Act == MM::AfterSet )
  {
    string vNewFileName;
    Prop->Get( vNewFileName );
    FileName_ = vNewFileName;
    //SetFileName( vNewFileName );
    MMDragonfly_->LogComponentMessage( "After Set: " + vNewFileName );
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


void CConfigFileHandler::SetFileName( const std::string& FileName )
{
  RequestedFileName_ = FileName;
}

int CConfigFileHandler::TestFileName( const std::string& FileName ) const
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

int CConfigFileHandler::UpdateFileName()
{
  return UpdateFileName( RequestedFileName_ );
}

int CConfigFileHandler::UpdateFileName( const std::string& FileName )
{
  MMDragonfly_->LogComponentMessage( "ConfigFileHandler: Updating file name [ " + FileName + " ]" );
  bool vNewFileExists = false;
  bool vNewFileReadWriteAcccess = false;
  fstream vNewFile( FileName, ios_base::in );
  if ( vNewFile.is_open() )
  {
    vNewFile.close();
    vNewFileExists = true;
  }
  vNewFile.open( FileName, ios_base::app );
  if ( vNewFile.is_open() )
  {
    vNewFile.close();
    vNewFileReadWriteAcccess = true;
  }
  vNewFileReadWriteAcccess = true;
  if ( !vNewFileReadWriteAcccess )
  {
    return ERR_CONFIGFILEIO_ERROR;
  }

  FileName_ = FileName;
  if (vNewFileExists )
  {
    MMDragonfly_->LogComponentMessage( "ConfigFileHandler: Loading newly set config file since it exists already" );
    // If new file exists => load configuration from new file
    LoadConfig();
  }
  else
  {
    // If new file doesn't exists => save current configuration in the new file
    MMDragonfly_->LogComponentMessage( "ConfigFileHandler: Saving current config in newly selected file since it doesn't exist" );
    SaveConfig();
  }

  return DEVICE_OK;
}

int CConfigFileHandler::LoadConfig()
{
  int vRet = TestFileName( FileName_ );
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
      // buffer contains of format: key1=value1\0key2=value2
      // walk the buffer extracting values
      const char* vSubstr = vBuffer;
      while ( '\0' != *vSubstr )
      {
        // length of key-value pair substring
        size_t vSubstrLen = strlen( vSubstr );

        // split substring on '=' char
        const char* vValuePos = strchr( vSubstr, '=' );
        if ( NULL != vValuePos )
        {
          // todo: remove "magic number" for buffer size 
          char vKey[256] = "";
          char vValue[256] = "";
          strncpy( vKey, vSubstr, vValuePos - vSubstr );
          strncpy( vValue, vValuePos + 1, vSubstrLen - ( vValuePos - vSubstr ) );
          Config_[vKey] = vValue;
        }

        // jump over the current substring plus its null
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
    //WritePrivateProfileString( _T( "Section 2" ), _T( "Key 2" ), _T( "Hello World" ), _T( "C:\\test.ini" ) );
    if ( !WritePrivateProfileString( g_SectionName, vConfigIt->first.c_str(), vConfigIt->second.c_str(), FileName_.c_str() ) )
    {
      MMDragonfly_->LogComponentMessage( "Failed to write data to Dragonfly ini file. Error returned: " + to_string( GetLastError() ) );
    }
    ++vConfigIt;
  }
}
