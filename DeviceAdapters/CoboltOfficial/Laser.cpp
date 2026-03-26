///////////////////////////////////////////////////////////////////////////////
// FILE:       Laser.cpp
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
// AUTHORS:       Lukas Kalinski / lukas.kalinski@coboltlasers.com (2025)
//

#include <assert.h>

#include "Laser.h"

#include "LaserDriver.h"
#include "LaserStateProperty.h"
#include "CustomizableEnumerationProperty.h"
#include "EnumerationProperty.h"
#include "NoShutterCommandLegacyFix.h"
#include "LaserShutterProperty.h"
#include "MutableDeviceProperty.h"
#include "ImmutableEnumerationProperty.h"
#include "StaticStringProperty.h"

using namespace std;
using namespace cobolt;

const std::string Laser::Milliamperes = "mA";
const std::string Laser::Amperes = "A";
const std::string Laser::Milliwatts = "mW";
const std::string Laser::Watts = "W";

const std::string Laser::EnumerationItem_On = "on";
const std::string Laser::EnumerationItem_Off = "off";
const std::string Laser::EnumerationItem_Enabled = "enabled";
const std::string Laser::EnumerationItem_Disabled = "disabled";

const std::string Laser::EnumerationItem_RunMode_ConstantCurrent = "Constant Current";
const std::string Laser::EnumerationItem_RunMode_ConstantPower = "Constant Power";
const std::string Laser::EnumerationItem_RunMode_Modulation = "Modulation";

int Laser::NextId__ = 1;

Laser::~Laser()
{
    const bool pausedPropertyIsPublic = ( shutter_ != NULL && properties_.find( shutter_->GetName() ) != properties_.end() );

    if ( !pausedPropertyIsPublic ) {
        delete shutter_;
    }

    for ( PropertyIterator it = GetPropertyIteratorBegin(); it != GetPropertyIteratorEnd(); it++ ) {
        delete it->second;
    }

    properties_.clear();
}

const std::string& Laser::GetId() const
{
    return id_;
}

const std::string& Laser::GetName() const
{
    return name_;
}

void Laser::SetOn( const bool on )
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

void Laser::SetShutterOpen( const bool open )
{
    if ( shutter_ == NULL ) {

        Logger::Instance()->LogError( "Laser::SetShutterOpen(): Shutter not available" );
        return;
    }

    shutter_->SetValue( open ? LaserShutterProperty::Value_Open : LaserShutterProperty::Value_Closed );
}

bool Laser::IsShutterOpen() const
{
    if ( shutter_ == NULL ) {

        Logger::Instance()->LogError( "Laser::IsShutterOpen(): Shutter not available" );
        return false;
    }

    return ( shutter_->IsOpen() );
}

Property* Laser::GetProperty( const std::string& name ) const
{
    return properties_.at( name );
}

Property* Laser::GetProperty( const std::string& name )
{
    return properties_[ name ];
}

Laser::PropertyIterator Laser::GetPropertyIteratorBegin()
{
    return properties_.begin();
}

Laser::PropertyIterator Laser::GetPropertyIteratorEnd()
{
    return properties_.end();
}

bool Laser::IsShutterEnabled() const
{
    return true;
}

Laser::Laser( const std::string& name, const std::string& wavelength, LaserDriver* driver ) :
    id_( std::to_string( (long double)NextId__++ ) ),
    name_( name ),
    laserDriver_( driver ),
    currentUnit_( "[?]" ),
    powerUnit_( "[?]" ),
    laserOnOffProperty_( NULL ),
    shutter_( NULL ),
    laserStateProperty_( NULL )
{
}

void Laser::CreateAdapterVersionProperty()
{
    RegisterPublicProperty( new StaticStringProperty( "Adapter Version", COBOLT_MM_DRIVER_VERSION ) );
}

void Laser::CreateAnalogImpedanceProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Impedance", laserDriver_, "system:input:analog:impedance?" );

    property->RegisterEnumerationItem( "HIGH", "system:input:analog:impedance HIGH", "1 kOhm" );
    property->RegisterEnumerationItem( "LOW", "system:input:analog:impedance LOW", "50 Ohm" );

    RegisterPublicProperty( property );
}

void Laser::CreateAnalogModulationProperty()
{
    // TODO: Add for cm too
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Modulation", laserDriver_, "laser:pm:analog:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:pm:analog:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:pm:analog:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void Laser::CreateAutostartControlProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Laser Control", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "autostart:abort", "Abort" );
    property->RegisterEnumerationItem( "__", "autostart:restart", "Restart" );
    RegisterPublicProperty( property );
}

void Laser::CreateCcCurrentSetpointProperty()
{
    std::string maxCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cc:current:setpoint? max", &maxCurrentSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCcCurrentSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxCurrentSetpoint = atof( maxCurrentSetpointResponse.c_str() );

    MutableDeviceProperty* property = new MutableNumericProperty<double>(
        "CC: Current Setpoint [" + currentUnit_ + "]", laserDriver_, "laser:cc:current:setpoint?", "laser:cc:current:setpoint", 0.0f, maxCurrentSetpoint );
    RegisterPublicProperty( property );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void Laser::CreateClearFaultProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Clear Fault", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "fault:clear", "Clear Fault" );
    RegisterPublicProperty( property );
}

void Laser::CreateCmCurrentHighSetpointProperty()
{
    std::string maxCmCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cm:current:high:setpoint? max", &maxCmCurrentSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCmCurrentHighSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxCmCurrentSetpoint = atof( maxCmCurrentSetpointResponse.c_str() );

    RegisterPublicProperty( new MutableNumericProperty<double>(
        "Current Mod: High Current Setpoint [" + currentUnit_ + "]", laserDriver_, "laser:cm:current:high:setpoint?", "laser:cm:current:high:setpoint", 0, maxCmCurrentSetpoint ) );
}

void Laser::CreateCpPowerSetpointProperty()
{
    std::string maxPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cp:power:setpoint? max", &maxPowerSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCpPowerSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxPowerSetpoint = atof( maxPowerSetpointResponse.c_str() );

    MutableDeviceProperty* property = new MutableNumericProperty<double>(
        "CP: Power Setpoint [" + powerUnit_ + "]", laserDriver_, "laser:cp:power:setpoint?", "laser:cp:power:setpoint", 0.0f, maxPowerSetpoint );
    RegisterPublicProperty( property );
}

void Laser::CreateCurrentReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Current Reading [" + currentUnit_ + "]", laserDriver_, "laser:current:reading?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void Laser::CreateDigitalModulationProperty()
{
    // TODO: Add for cm too
    // TODO NOW: Merge digital/analog modulation for pm and cm into one
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Digital Modulation", laserDriver_, "laser:pm:digital:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:pm:digital:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:pm:digital:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void Laser::CreateFaultProperty()
{
    DeviceProperty* faultProperty = new DeviceProperty( Property::Stereotype::String, "Laser Fault", laserDriver_, "fault?" );
    faultProperty->SetCaching( false );
    RegisterPublicProperty( faultProperty );
}

void Laser::CreateFirmwareVersionProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Firmware Version", laserDriver_, "gfv?" ) );
}

void Laser::CreateKeyswitchProperty()
{
    ImmutableEnumerationProperty* property = new ImmutableEnumerationProperty( "Keyswitch", laserDriver_, "gkses?" );

    property->RegisterEnumerationItem( "0", "Disabled" );
    property->RegisterEnumerationItem( "1", "Enabled" );

    RegisterPublicProperty( property );
}

void Laser::CreateLaserOnOffProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Laser Status", laserDriver_, "l?" );

    property->RegisterEnumerationItem( "0", "abort", EnumerationItem_Off );
    property->RegisterEnumerationItem( "1", "restart", EnumerationItem_On );
    property->SetCaching( false );

    RegisterPublicProperty( property );
    laserOnOffProperty_ = property;
}

void Laser::CreateLaserStateProperty()
{
    laserStateProperty_ = new DeviceProperty( Property::Stereotype::String, "Laser State", laserDriver_, "state?" );
    laserStateProperty_->SetCaching( false );
    RegisterPublicProperty( laserStateProperty_ );
}

void Laser::CreateModelProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Model", laserDriver_, "glm?" ) );
}

void Laser::CreateModulationInputVoltageMaxProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Modulation Input Voltage Max", laserDriver_, "system:input:analog:voltage:range:max?" );

    property->RegisterEnumerationItem( "1", "system:input:analog:voltage:range:max 1", "1 V" );
    property->RegisterEnumerationItem( "5", "system:input:analog:voltage:range:max 5", "5 V" );

    RegisterPublicProperty( property );
}

void Laser::CreateNameProperty()
{
    RegisterPublicProperty( new StaticStringProperty( "Name", this->GetName() ) );
}

void Laser::CreateOperatingHoursProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Operating Hours", laserDriver_, "hrs?" ) );
}

void Laser::CreatePmPowerSetpointProperty()
{
    std::string maxPmPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:pm:power:setpoint? max", &maxPmPowerSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreatePmPowerSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxPmPowerSetpoint = atof( maxPmPowerSetpointResponse.c_str() );

    RegisterPublicProperty( new MutableNumericProperty<double>(
        "Power Mod: Power Setpoint [" + powerUnit_ + "]", laserDriver_, "laser:pm:power:setpoint?", "laser:pm:power:setpoint", 0, maxPmPowerSetpoint ) );
}

void Laser::CreatePowerReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Power Reading [" + powerUnit_ + "]", laserDriver_, "laser:power:reading?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void Laser::CreateRunmodeProperty()
{
    EnumerationProperty* property = new EnumerationProperty( "Runmode", laserDriver_, "laser:runmode" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void Laser::CreateSerialNumberProperty()
{
    RegisterPublicProperty( new DeviceProperty( Property::Stereotype::String, "Serial Number", laserDriver_, "gsn?" ) );
}

void Laser::CreateShutterProperty()
{
    shutter_ = new LaserShutterProperty( "Emission Status", laserDriver_, this );
    RegisterPublicProperty( shutter_ );
}

void Laser::CreateWavelengthProperty( const std::string& wavelength )
{
    RegisterPublicProperty( new StaticStringProperty( "Wavelength", wavelength ) );
}

bool Laser::IsShutterCommandSupported() const
{
    return true;
}

bool Laser::IsInCdrhMode() const
{
    std::string response;

    if ( response.empty() ) {
        laserDriver_->SendCommand( "autostart:enabled?", &response );
    }

    return ( response == "1" );
}

void Laser::RegisterPublicProperty( Property* property )
{
    assert( property != NULL );
    properties_[ property->GetName() ] = property;
    Logger::Instance()->LogMessage( "Registered property: " + property->GetName(), true );
}

double Laser::MaxCurrentSetpoint()
{
    std::string maxCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "gmlc?", &maxCurrentSetpointResponse ) != return_code::ok ) {

        Logger::Instance()->LogError( "Laser::MaxCurrentSetpoint(): Failed to retrieve max current sepoint" );
        return 0.0f;
    }

    return atof( maxCurrentSetpointResponse.c_str() );
}

double Laser::MaxPowerSetpoint()
{
    std::string maxPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "gmlp?", &maxPowerSetpointResponse ) != return_code::ok ) {

        Logger::Instance()->LogError( "Laser::MaxPowerSetpoint(): Failed to retrieve max power sepoint" );
        return 0.0f;
    }

    return atof( maxPowerSetpointResponse.c_str() );
}
