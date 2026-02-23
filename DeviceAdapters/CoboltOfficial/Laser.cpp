///////////////////////////////////////////////////////////////////////////////
// FILE:       LysaLaser.cpp
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

#include "LysaLaser.h"

#include "LaserDriver.h"
#include "LaserStateProperty.h"
#include "CustomizableEnumerationProperty.h"
#include "EnumerationProperty.h"
#include "NoShutterCommandLegacyFix.h"

using namespace std;
using namespace cobolt;

LysaLaser::LysaLaser( const std::string& name, const std::string& wavelength, LaserDriver* driver ) :
    Laser( name, driver ),
    laserStateProperty_( NULL )
{
    currentUnit_ = Milliamperes;
    powerUnit_ = Milliwatts;

    CreateNameProperty();
    CreateWavelengthProperty( wavelength );
    CreateModelProperty();
    CreateSerialNumberProperty();
    CreateFirmwareVersionProperty();
    CreateAdapterVersionProperty();
    CreateOperatingHoursProperty();
    CreateKeyswitchProperty();
    CreateLaserStateProperty();
    CreateFaultProperty();

    CreatePropertyGroup( "Control" );
    CreateClearFaultProperty();
    CreateRestartProperty();
    CreateAbortProperty();
    CreateShutterProperty();
    CreateRunmodeProperty();

    CreatePropertyGroup( "Readings" );
    CreatePowerReadingProperty();
    CreateCurrentReadingProperty();

    CreatePropertyGroup( "Runmode: Constant Power" );
    CreateCpPowerSetpointProperty();

    CreatePropertyGroup( "Runmode: Constant Current" );
    CreateCcCurrentSetpointProperty();

    CreatePropertyGroup( "Runmode: Power Modulation" );
    CreatePmPowerSetpointProperty();
    CreatePmDigitalModulationProperty();
    CreatePmAnalogModulationProperty();

    CreatePropertyGroup( "Runmode: Current Modulation" );
    CreateCmCurrentHighSetpointProperty();
    CreateCmDigitalModulationProperty();
    CreateCmAnalogModulationProperty();

    CreatePropertyGroup( "Modulation Settings" );
    CreateAnalogImpedanceProperty();
    CreateModulationInputVoltageMaxProperty();
}

void LysaLaser::CreateAnalogImpedanceProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Impedance", laserDriver_, "system:input:analog:impedance?" );

    property->RegisterEnumerationItem( "HIGH", "system:input:analog:impedance HIGH", "1 kOhm" );
    property->RegisterEnumerationItem( "LOW", "system:input:analog:impedance LOW", "50 Ohm" );

    RegisterPublicProperty( property );
}

void LysaLaser::CreateModulationInputVoltageMaxProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Modulation Input Voltage Max", laserDriver_, "system:input:analog:voltage:range:max?" );

    property->RegisterEnumerationItem( "1", "system:input:analog:voltage:range:max 1", "1 V" );
    property->RegisterEnumerationItem( "5", "system:input:analog:voltage:range:max 5", "5 V" );

    RegisterPublicProperty( property );
}

void LysaLaser::CreateShutterProperty()
{
    shutter_ = new LaserShutterProperty( "Emission Status", laserDriver_, this );
    RegisterPublicProperty( shutter_ );
}

bool LysaLaser::IsShutterEnabled() const
{
    return true;
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void LysaLaser::CreateClearFaultProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Clear Fault", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "fault:clear", "Clear Fault" );
    RegisterPublicProperty( property );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void LysaLaser::CreateRestartProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Restart", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "autostart:restart", "Restart" );
    RegisterPublicProperty( property );
}

/**
 * This is a hacky function to get around the problem of not having any button feature available in the Property Browser.
 */
void LysaLaser::CreateAbortProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Abort", laserDriver_, "?" );
    property->RegisterEnumerationItem( "OK", "?", "---" ); // Without this we get an error when adding a new laser.
    property->RegisterEnumerationItem( "__", "autostart:abort", "Abort" );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateLaserStateProperty()
{
    laserStateProperty_ = new DeviceProperty( Property::Stereotype::String, "Laser State", laserDriver_, "state?" );
    laserStateProperty_->SetCaching( false );
    RegisterPublicProperty( laserStateProperty_ );
}

void LysaLaser::CreateFaultProperty()
{
    DeviceProperty* faultProperty = new DeviceProperty( Property::Stereotype::String, "Laser Fault", laserDriver_, "fault?" );
    faultProperty->SetCaching( false );
    RegisterPublicProperty( faultProperty );
}

void LysaLaser::CreateCurrentReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Current Reading [" + currentUnit_ + "]", laserDriver_, "laser:current:reading?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void LysaLaser::CreatePowerReadingProperty()
{
    DeviceProperty* property = new DeviceProperty( Property::Stereotype::Float, "Power Reading [" + powerUnit_ + "]", laserDriver_, "laser:power:reading?" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateCcCurrentSetpointProperty()
{
    std::string maxCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cc:current:setpoint? max", &maxCurrentSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCcCurrentSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxCurrentSetpoint = atof( maxCurrentSetpointResponse.c_str() );

    MutableDeviceProperty* property = new MutableNumericProperty<double>( 
        "Current Setpoint [" + currentUnit_ + "]", laserDriver_, "laser:cc:current:setpoint?", "laser:cc:current:setpoint", 0.0f, maxCurrentSetpoint );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateCpPowerSetpointProperty()
{
    std::string maxPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cp:power:setpoint? max", &maxPowerSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCpPowerSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxPowerSetpoint = atof( maxPowerSetpointResponse.c_str() );

    MutableDeviceProperty* property = new MutableNumericProperty<double>(
        "Power Setpoint [" + powerUnit_ + "]", laserDriver_, "laser:cp:power:setpoint?", "laser:cp:power:setpoint", 0.0f, maxPowerSetpoint );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateRunmodeProperty()
{
    EnumerationProperty* property = new EnumerationProperty( "Runmode", laserDriver_, "laser:runmode" );
    property->SetCaching( false );
    RegisterPublicProperty( property );
}

void LysaLaser::CreatePmPowerSetpointProperty()
{
    std::string maxPmPowerSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:pm:power:setpoint? max", &maxPmPowerSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreatePmPowerSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxPmPowerSetpoint = atof( maxPmPowerSetpointResponse.c_str() );

    RegisterPublicProperty( new MutableNumericProperty<double>(
        "Power Setpoint", laserDriver_, "laser:pm:power:setpoint?", "laser:pm:power:setpoint", 0, maxPmPowerSetpoint ) );
}

void LysaLaser::CreateCmCurrentHighSetpointProperty()
{
    std::string maxCmCurrentSetpointResponse;
    if ( laserDriver_->SendCommand( "laser:cm:current:high:setpoint? max", &maxCmCurrentSetpointResponse ) != return_code::ok ) {
        Logger::Instance()->LogError( "Laser::CreateCmCurrentHighSetpointProperty(): Failed to retrieve max power sepoint" );
        return;
    }

    const double maxCmCurrentSetpoint = atof( maxCmCurrentSetpointResponse.c_str() );

    RegisterPublicProperty( new MutableNumericProperty<double>(
        "High Current Setpoint", laserDriver_, "laser:cm:current:high:setpoint?", "laser:cm:current:high:setpoint", 0, maxCmCurrentSetpoint ) );
}

void LysaLaser::CreatePmDigitalModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Digital Modulation", laserDriver_, "laser:pm:digital:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:pm:digital:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:pm:digital:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void LysaLaser::CreatePmAnalogModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Modulation", laserDriver_, "laser:pm:analog:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:pm:analog:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:pm:analog:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateCmDigitalModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Digital Modulation", laserDriver_, "laser:cm:digital:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:cm:digital:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:cm:digital:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}

void LysaLaser::CreateCmAnalogModulationProperty()
{
    CustomizableEnumerationProperty* property = new CustomizableEnumerationProperty( "Analog Modulation", laserDriver_, "laser:cm:analog:enabled?" );
    property->RegisterEnumerationItem( "0", "laser:cm:analog:enabled 0", EnumerationItem_Disabled );
    property->RegisterEnumerationItem( "1", "laser:cm:analog:enabled 1", EnumerationItem_Enabled );
    RegisterPublicProperty( property );
}
