///////////////////////////////////////////////////////////////////////////////
// FILE:       LegacyLaser.cpp
// PROJECT:    MicroManager
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:
// Cobolt Lasers Controller Adapter
//
// COPYRIGHT:     Cobolt AB, Stockholm, 2020
//                All rights reserved
//
// LICENSE:       MIT
//                Permission is hereby granted, free of charge, to any person obtaining a
//                copy of this software and associated documentation files( the "Software" ),
//                to deal in the Software without restriction, including without limitation the
//                rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//                sell copies of the Software, and to permit persons to whom the Software is
//                furnished to do so, subject to the following conditions:
//                
//                The above copyright notice and this permission notice shall be included in all
//                copies or substantial portions of the Software.
//
//                THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//                INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//                PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//                HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//                OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//                SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// CAUTION:       Use of controls or adjustments or performance of any procedures other than those
//                specified in owner's manual may result in exposure to hazardous radiation and
//                violation of the CE / CDRH laser safety compliance.
//
// AUTHORS:       Lukas Kalinski / lukas.kalinski@coboltlasers.com (2020)
//

#include <assert.h>
#include <algorithm>
#include "LegacyLaser.h"
#include "Logger.h"

#include "LaserDriver.h"
#include "StaticStringProperty.h"
#include "DeviceProperty.h"
#include "ImmutableEnumerationProperty.h"
#include "LaserStateProperty.h"
#include "MutableDeviceProperty.h"
#include "CustomizableEnumerationProperty.h"
#include "MutableNumericProperty.h"
#include "LaserShutterProperty.h"
#include "NoShutterCommandLegacyFix.h"

using namespace std;
using namespace cobolt;

LegacyLaser::LegacyLaser( const std::string& name, LaserDriver* driver ) :
    Laser( name, "", driver),
    laserStatePropertyOld_( NULL )
{
}

LegacyLaser::~LegacyLaser()
{
}

const std::string& LegacyLaser::GetId() const
{
    return id_;
}

const std::string& LegacyLaser::GetName() const
{
    return name_;
}

void LegacyLaser::SetOn( const bool on )
{
    // Reset shutter on laser on/off:
    SetShutterOpen( false );

    if ( laserOnOffProperty_ != NULL && false ) { // TODO: replace 'false' with 'autostart disabled'
        
        laserOnOffProperty_->SetValue( ( on ? EnumerationItem_On : EnumerationItem_Off ) );

    } else {
        
        if ( on ) {
            laserDriver_->SendCommand( "restart" );
        } else {
            laserDriver_->SendCommand( "abort" );
        }
    }
}

void LegacyLaser::SetShutterOpen( const bool open )
{
    if ( shutter_ == NULL ) {

        Logger::Instance()->LogError( "Laser::SetShutterOpen(): Shutter not available" );
        return;
    }

    shutter_->SetValue( open ? LaserShutterProperty::Value_Open : LaserShutterProperty::Value_Closed );
}

bool LegacyLaser::IsShutterEnabled() const
{
    if ( laserOnOffProperty_ != NULL ) {

        // Always enabled if open:
        return ( laserOnOffProperty_->GetValue() == EnumerationItem_On );

    } else if ( laserStatePropertyOld_ != NULL ) {
        
        return ( laserStatePropertyOld_->AllowsShutter() );
    }
    
    Logger::Instance()->LogError( "Laser::IsShutterEnabled(): Expected properties were not initialized" );
    return false;
}

bool LegacyLaser::IsShutterOpen() const
{
    if ( shutter_ == NULL ) {

        Logger::Instance()->LogError( "Laser::IsShutterOpen(): Shutter not available" );
        return false;
    }

    return ( shutter_->IsOpen() );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void LegacyLaser::CreateAutostartControlProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Laser Control", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "abort", "Abort" );
    property->RegisterEnumerationItem( "__", "restart", "Restart" );
    RegisterPublicProperty( property );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void LegacyLaser::CreateClearFaultProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Clear Fault", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "cf", "Clear Fault" );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateNameProperty()
{
    RegisterPublicProperty( new StaticStringProperty( "Name", this->GetName() ) );
}

void LegacyLaser::CreateModelProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Model", laserDriver_, "glm?") );
}

void LegacyLaser::CreateWavelengthProperty( const std::string& wavelength)
{
    RegisterPublicProperty( new StaticStringProperty( "Wavelength", wavelength ) );
}

void LegacyLaser::CreateKeyswitchProperty()
{
    ImmutableEnumerationProperty* property = new ImmutableEnumerationProperty( "Keyswitch", laserDriver_, "gkses?" );

    property->RegisterEnumerationItem( "0", "Disabled" );
    property->RegisterEnumerationItem( "1", "Enabled" );
    
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateSerialNumberProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Serial Number", laserDriver_, "gsn?") );
}

void LegacyLaser::CreateFirmwareVersionProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Firmware Version", laserDriver_, "gfv?") );
}

void LegacyLaser::CreateAdapterVersionProperty()
{
    RegisterPublicProperty( new StaticStringProperty( "Adapter Version", COBOLT_MM_DRIVER_VERSION ) );
}

void LegacyLaser::CreateOperatingHoursProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Operating Hours", laserDriver_, "hrs?") );
}

void LegacyLaser::CreateCcCurrentSetpointProperty()
{
    CreateCcCurrentSetpointProperty( "gdsn?", "sdsn" );
}

void LegacyLaser::CreateCcCurrentSetpointProperty( const std::string& getPersistedDataCommand, const std::string& setPersistedDataCommand )
{
    MutableDeviceProperty* property;
   
    if ( IsShutterCommandSupported() || !IsInCdrhMode() ) {
        property = new MutableNumericProperty<double>( "Current Setpoint [" + currentUnit_ + "]", laserDriver_, "glc?", "slc", 0.0f, MaxCurrentSetpoint() );
    } else {
        property = new legacy::no_shutter_command::LaserCurrentProperty(
            "Current Setpoint [" + currentUnit_ + "]",
            laserDriver_,
            "glc?",
            "slc",
            0.0f,
            MaxCurrentSetpoint(),
            this,
            getPersistedDataCommand,
            setPersistedDataCommand );
    }

    RegisterPublicProperty( property );
}

void LegacyLaser::CreateCurrentReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Current Reading [" + currentUnit_ + "]", laserDriver_, "i?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateCpPowerSetpointProperty()
{
    MutableDeviceProperty* property = new MutableNumericProperty<double>( "Power Setpoint [" + powerUnit_ + "]", laserDriver_, "glp?", "slp", 0.0f, MaxPowerSetpoint() );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreatePowerReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Power Reading [" + powerUnit_ + "]", laserDriver_, "pa?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateLaserOnOffProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Laser Status", laserDriver_, "l?" );

    property->RegisterEnumerationItem( "0", "abort", EnumerationItem_Off );
    property->RegisterEnumerationItem( "1", "restart", EnumerationItem_On );
    property->SetCaching( false );
    
    RegisterPublicProperty( property );
    laserOnOffProperty_ = property;
}

void LegacyLaser::CreateShutterProperty( std::string saveCmd, std::string readCmd )
{
    if ( IsShutterCommandSupported() ) {
        shutter_ = new LaserShutterProperty( "Emission Status", laserDriver_, this );
    } else {
        // From now on we use the same shutter solution regardless of oem/cdrh:
        shutter_ = new legacy::no_shutter_command::LaserShutterPropertyCdrh( "Emission Status", laserDriver_, this, readCmd, saveCmd );
    }
    
    RegisterPublicProperty( shutter_ );
}

void LegacyLaser::CreateDigitalModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Digital Modulation", laserDriver_, "gdmes?" );
    property->RegisterEnumerationItem( "0", "sdmes 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "sdmes 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateAnalogModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Modulation", laserDriver_,  "games?" );
    property->RegisterEnumerationItem( "0", "sames 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "sames 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreatePmPowerSetpointProperty()
{
    std::string maxModulationPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "gmlp?", &maxModulationPowerSetpointResponse ) != return_code::ok ) {

        Logger::Instance()->LogError( "Laser::CreateCpPowerSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }
    
    const double maxModulationPowerSetpoint = atof( maxModulationPowerSetpointResponse.c_str() );
    
    RegisterPublicProperty( new MutableNumericProperty<double>( "Modulation Power Setpoint", laserDriver_, "glmp?", "slmp", 0, maxModulationPowerSetpoint ) );
}

void LegacyLaser::CreateAnalogImpedanceProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Impedance", laserDriver_, "galis?" );
    
    property->RegisterEnumerationItem( "0", "salis 0", "1 kOhm" );
    property->RegisterEnumerationItem( "1", "salis 1", "50 Ohm" );

    RegisterPublicProperty( property );
}

void LegacyLaser::CreateModulationCurrentLowSetpointProperty()
{
    MutableDeviceProperty* property;
    property = new MutableNumericProperty<double>( "Modulation Low Current Setpoint [" + currentUnit_ + "]", laserDriver_, "glth?", "slth", 0.0f, MaxCurrentSetpoint() );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateModulationCurrentHighSetpointProperty()
{
    MutableDeviceProperty* property;
    property = new MutableNumericProperty<double>( "Modulation Low Current Setpoint [" + currentUnit_ + "]", laserDriver_, "gmc?", "smc", 0.0f, MaxCurrentSetpoint() );
    RegisterPublicProperty( property );
}

void LegacyLaser::CreateModulationHighPowerSetpointProperty()
{
    MutableDeviceProperty* property = new MutableNumericProperty<double>( "Modulation Power Setpoint [" + powerUnit_ + "]", laserDriver_, "glmp?", "slmp", 0.0f, MaxPowerSetpoint() );
    RegisterPublicProperty( property );
}

bool LegacyLaser::IsShutterCommandSupported() const // TODO: Split into IsShutterCommandSupported() and IsPauseCommandSupported()
{
    std::string response;

    if ( response.empty() ) {
        laserDriver_->SendCommand( "l0r", &response );
    }

    return (!( response.find( "Invalid" ) != std::string::npos || response.find("illegal") != std::string::npos));
}

bool LegacyLaser::IsInCdrhMode() const
{
    std::string response;

    if ( response.empty() ) {
        laserDriver_->SendCommand( "gas?", &response );
    }

    return ( response == "1" );
}

void LegacyLaser::RegisterPublicProperty( Property* property )
{
    assert( property != NULL );
    properties_[ property->GetName() ] = property;
}

double LegacyLaser::MaxCurrentSetpoint()
{
    std::string maxCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "gmlc?", &maxCurrentSetpointResponse ) != return_code::ok ) {

        Logger::Instance()->LogError( "Laser::MaxCurrentSetpoint(): Failed to retrieve max current sepoint" );
        return 0.0f;
    }
    
    return atof( maxCurrentSetpointResponse.c_str() );
}

double LegacyLaser::MaxPowerSetpoint()
{
    std::string maxPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "gmlp?", &maxPowerSetpointResponse ) != return_code::ok ) {

        Logger::Instance()->LogError( "Laser::MaxPowerSetpoint(): Failed to retrieve max power sepoint" );
        return 0.0f;
    }

    return atof( maxPowerSetpointResponse.c_str() );
}
