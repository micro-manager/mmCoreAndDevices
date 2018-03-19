#include "TIRF.h"
#include "Dragonfly.h"
#include "TIRFModeIntSubProperty.h"
#include "TIRFModeFloatSubProperty.h"

#include <stdexcept>

using namespace std;

const char* const g_Undefined = "undefined";

const char* const g_TIRFModePropertyName = "TIRF Mode";
const char* const g_TIRFModePenetration = "Penetration";
const char* const g_TIRFModeHILO = "HiLo Illumination";
const char* const g_TIRFModeCritical = "Critical Angle";

const char* const g_MagnificationPropertyName = "TIRF | Magnification";
const char* const g_NumericalAperturePropertyName = "TIRF | Numerical Aperture";
const char* const g_RefractiveIndexPropertyName = "TIRF | Refractive Index";
const char* const g_ScopePropertyName = "TIRF | Scope Type";
const char* const g_ScopeLeica = "Leica";
const char* const g_ScopeNikon = "Nikon";
const char* const g_ScopeOlympus = "Olympus";
const char* const g_ScopeZeiss = "Zeiss";

CTIRF::CTIRF( ITIRFInterface* TIRFInterface, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly )
  : MMDragonfly_( MMDragonfly ),
  ConfigFileHandler_( ConfigFileHandler ),
  TIRFInterface_( TIRFInterface ),
  PenetrationProperty_( nullptr ),
  PenetrationWrapper_( nullptr ),
  HILOObliqueAngleProperty_( nullptr ),
  HILOObliqueAngleWrapper_( nullptr ),
  CriticalAngleOffsetProperty_( nullptr ),
  CriticalAngleOffsetWrapper_( nullptr )
{
  // Retrieve current values from the device
  if ( !TIRFInterface_->GetOpticalPathway( &Magnification_, &NumericalAperture_, &RefractiveIndex_, &ScopeID_ ) )
  {
    throw runtime_error( "Failed to read the current TIRF optical pathway" );
  }

  int vTIRFMode;
  if ( !TIRFInterface_->GetTIRFMode( &vTIRFMode ) )
  {
    throw runtime_error( "Failed to read the current TIRF mode" );
  }

  const char* vTIRFModeName;
  if ( !GetTIRFModeNameFromIndex( vTIRFMode, &vTIRFModeName ) )
  {
    throw runtime_error( "Invalid TIRF mode found on initialisation" );
  }

  // Create MM property for TIRF mode
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRF::OnTIRFModeChange );
  int vRet = MMDragonfly_->CreateStringProperty( g_TIRFModePropertyName, vTIRFModeName, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + string( g_TIRFModePropertyName ) + " property" );
  }
  MMDragonfly_->AddAllowedValue( g_TIRFModePropertyName, g_TIRFModePenetration );
  MMDragonfly_->AddAllowedValue( g_TIRFModePropertyName, g_TIRFModeHILO );
  MMDragonfly_->AddAllowedValue( g_TIRFModePropertyName, g_TIRFModeCritical );

  // Property for Penetration
  PenetrationWrapper_ = new CPenetrationWrapper( TIRFInterface_ );
  PenetrationProperty_ = new CTIRFModeIntSubProperty( PenetrationWrapper_, ConfigFileHandler_, MMDragonfly_, "TIRF |  Penetration (nm)" );
  // Property for Oblique Angle
  HILOObliqueAngleWrapper_ = new CHILOObliqueAngleWrapper( TIRFInterface_ );
  HILOObliqueAngleProperty_ = new CTIRFModeFloatSubProperty( HILOObliqueAngleWrapper_, ConfigFileHandler_, MMDragonfly_, "TIRF |  HiLo Oblique Angle (deg)" );
  // Property for Offset
  CriticalAngleOffsetWrapper_ = new COffsetWrapper( TIRFInterface_ );
  CriticalAngleOffsetProperty_ = new CTIRFModeIntSubProperty( CriticalAngleOffsetWrapper_, ConfigFileHandler_, MMDragonfly_, "TIRF |  Critical Angle Offset" );
  
  // Set TIRF sub properties read/write status based on TIRF mode
  UpdateTIRFModeSelection( vTIRFModeName );

  // Create MM properties for optical pathway
  vRet = MMDragonfly_->CreateIntegerProperty( g_MagnificationPropertyName, Magnification_, true );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_MagnificationPropertyName ) + " property" );
    return;
  }
  vRet = MMDragonfly_->CreateFloatProperty( g_NumericalAperturePropertyName, NumericalAperture_, true );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_NumericalAperturePropertyName ) + " property" );
    return;
  }
  vRet = MMDragonfly_->CreateFloatProperty( g_RefractiveIndexPropertyName, RefractiveIndex_, true );
  if ( vRet != DEVICE_OK )
  {
    MMDragonfly_->LogComponentMessage( "Error creating " + string( g_RefractiveIndexPropertyName ) + " property" );
    return;
  }
  const char* vScopeName;
  if ( GetScopeNameFromIndex( ScopeID_, &vScopeName ) )
  {
    vRet = MMDragonfly_->CreateStringProperty( g_ScopePropertyName, vScopeName, true );
    if ( vRet != DEVICE_OK )
    {
      MMDragonfly_->LogComponentMessage( "Error creating " + string( g_ScopePropertyName ) + " property" );
      return;
    }
  }
  else
  {
    vRet = MMDragonfly_->CreateStringProperty( g_ScopePropertyName, "undefined", true );
    if ( vRet != DEVICE_OK )
    {
      MMDragonfly_->LogComponentMessage( "Error creating " + string( g_ScopePropertyName ) + " property" );
      return;
    }
    MMDragonfly_->LogComponentMessage( "Current TIRF scope invalid" );
  }
  MMDragonfly_->AddAllowedValue( g_ScopePropertyName, g_ScopeLeica );
  MMDragonfly_->AddAllowedValue( g_ScopePropertyName, g_ScopeNikon );
  MMDragonfly_->AddAllowedValue( g_ScopePropertyName, g_ScopeOlympus );
  MMDragonfly_->AddAllowedValue( g_ScopePropertyName, g_ScopeZeiss );
}

CTIRF::~CTIRF()
{
  delete PenetrationProperty_;
  delete PenetrationWrapper_;
  delete HILOObliqueAngleProperty_;
  delete HILOObliqueAngleWrapper_;
  delete CriticalAngleOffsetProperty_;
  delete CriticalAngleOffsetWrapper_;
}

int CTIRF::OnTIRFModeChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    string vRequestedMode;
    Prop->Get( vRequestedMode );
    int vTIRFModeIndex;
    if ( GetTIFRModeIndexFromName( vRequestedMode, &vTIRFModeIndex ) )
    {
      if ( TIRFInterface_->SetTIRFMode( vTIRFModeIndex ) )
      {
        UpdateTIRFModeSelection( vRequestedMode );
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Failed to set TIRF Mode [" + vRequestedMode + "]" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Requested TIRF Mode [" + vRequestedMode + "] is invalid. Ignoring request." );
      vRet = DEVICE_ERR;
    }
  }
  return vRet;
}

bool CTIRF::GetTIRFModeNameFromIndex( int TIRFModeIndex, const char** TIRFModName )
{
  bool vModeFound = false;
  switch ( TIRFModeIndex )
  {
  case 0:
    *TIRFModName = g_TIRFModePenetration;
    vModeFound = true;
    break;
  case 1:
    *TIRFModName = g_TIRFModeHILO;
    vModeFound = true;
    break;
  case 2:
    *TIRFModName = g_TIRFModeCritical;
    vModeFound = true;
    break;
  }
  return vModeFound;
}

bool CTIRF::GetTIFRModeIndexFromName( const string& TIRFModeName, int* TIRFModeIndex )
{
  bool vModeFound = false;
  if ( TIRFModeName == g_TIRFModePenetration )
  {
    *TIRFModeIndex = 0;
    vModeFound = true;
  }
  else if ( TIRFModeName == g_TIRFModeHILO )
  {
    *TIRFModeIndex = 1;
    vModeFound = true;
  }
  else if ( TIRFModeName == g_TIRFModeCritical )
  {
    *TIRFModeIndex = 2;
    vModeFound = true;
  }
  return vModeFound;
}

bool CTIRF::GetScopeNameFromIndex( int ScopeID, const char** ScopeName )
{
  bool vScopeFound = false;
  switch ( ScopeID )
  {
  case st_Leica:
    *ScopeName = g_ScopeLeica;
    vScopeFound = true;
    break;
  case st_Nikon:
    *ScopeName = g_ScopeNikon;
    vScopeFound = true;
    break;
  case st_Olympus:
    *ScopeName = g_ScopeOlympus;
    vScopeFound = true;
    break;
  case st_Zeiss:
    *ScopeName = g_ScopeZeiss;
    vScopeFound = true;
    break;
  }
  return vScopeFound;
}

bool CTIRF::GetScopeIndexFromName( const string& ScopeName, int* ScopeIndex )
{
  bool vScopeFound = false;
  if ( ScopeName == g_ScopeLeica )
  {
    *ScopeIndex = st_Leica;
    vScopeFound = true;
  }
  else if ( ScopeName == g_ScopeNikon )
  {
    *ScopeIndex = st_Nikon;
    vScopeFound = true;
  }
  else if ( ScopeName == g_ScopeOlympus )
  {
    *ScopeIndex = st_Olympus;
    vScopeFound = true;
  }
  else if ( ScopeName == g_ScopeZeiss )
  {
    *ScopeIndex = st_Zeiss;
    vScopeFound = true;
  }
  return vScopeFound;
}

void CTIRF::UpdateTIRFModeSelection( const string& TIRFModeName )
{
  if ( TIRFModeName == g_TIRFModePenetration )
  {
    PenetrationProperty_->ModeSelected( Penetration );
    HILOObliqueAngleProperty_->ModeSelected( Penetration );
    CriticalAngleOffsetProperty_->ModeSelected( Penetration );
  }
  else if ( TIRFModeName == g_TIRFModeHILO )
  {
    PenetrationProperty_->ModeSelected( HiLoObliqueAngle );
    HILOObliqueAngleProperty_->ModeSelected( HiLoObliqueAngle );
    CriticalAngleOffsetProperty_->ModeSelected( HiLoObliqueAngle );
  }
  else if ( TIRFModeName == g_TIRFModeCritical )
  {
    PenetrationProperty_->ModeSelected( CriticalAngle );
    HILOObliqueAngleProperty_->ModeSelected( CriticalAngle );
    CriticalAngleOffsetProperty_->ModeSelected( CriticalAngle );
  }
}