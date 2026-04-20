///////////////////////////////////////////////////////////////////////////////
// FILE:       Gen5Laser.cpp
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

#include "Gen5Laser.h"

#include "LaserDriver.h"
#include "LaserStateProperty.h"
#include "CustomizableEnumerationProperty.h"
#include "NoShutterCommandLegacyFix.h"
#include "ImmutableEnumerationProperty.h"

using namespace std;
using namespace cobolt;

Gen5Laser::Gen5Laser( const std::string& wavelength, LaserDriver* driver ) :
    LegacyLaser( "05 Laser", driver )
{
    currentUnit_ = Amperes;
    powerUnit_ = Watts;

    //CreateWavelengthProperty( wavelength );

    CreateNameProperty();
    CreateModelProperty();
    CreateSerialNumberProperty();
    CreateFirmwareVersionProperty();
    CreateOperatingHoursProperty();
    CreateKeyswitchProperty();
    CreateLaserStateProperty();
    CreateFaultProperty();

    // Laser Control Group
    CreateClearFaultProperty();
    CreateAutostartControlProperty();
    CreateShutterProperty( "sartn", "gartn?" );
    CreateRunmodeProperty();

    //// Readings Group
    CreatePowerReadingProperty();
    CreateCurrentReadingProperty();

    //// Runmode Control Group
    CreateCpPowerSetpointProperty();
    CreateCcCurrentSetpointProperty( "gartn?", "sartn" );
    CreateAdapterVersionProperty();
}

void Gen5Laser::CreateLaserStateProperty()
{
    if ( IsInCdrhMode() ) {

        laserStatePropertyOld_ = new LaserStateProperty( Property::Stereotype::String, "Laser State", laserDriver_, "gom?" );
    
        laserStatePropertyOld_->RegisterState( "0", "Off", false );
        laserStatePropertyOld_->RegisterState( "1", "Waiting for Temperatures", false );
        laserStatePropertyOld_->RegisterState( "2", "Waiting for Key", false );
        laserStatePropertyOld_->RegisterState( "3", "Warming Up", false );
        laserStatePropertyOld_->RegisterState( "4", "Completed", true );
        laserStatePropertyOld_->RegisterState( "5", "Fault", false );
        laserStatePropertyOld_->RegisterState( "6", "Aborted", false );
        laserStatePropertyOld_->RegisterState( "7", "Waiting for Remote", false );
        laserStatePropertyOld_->RegisterState( "8", "Standby", false );

    } else {

        laserStatePropertyOld_ = new LaserStateProperty( Property::Stereotype::String, "Laser State", laserDriver_, "l?" );

        laserStatePropertyOld_->RegisterState( "0", "Off", true );
        laserStatePropertyOld_->RegisterState( "1", "On", true );
    }

    laserStatePropertyOld_->SetCaching( false );

    RegisterPublicProperty( laserStatePropertyOld_ );
}

void Gen5Laser::CreateRunmodeProperty()
{
    CustomizableEnumerationProperty* property;

    if ( IsShutterCommandSupported() || !IsInCdrhMode() ) {
        property = new CustomizableEnumerationProperty( "Runmode", laserDriver_, "gam?" );
    } else {
        property = new legacy::no_shutter_command::LaserRunModeProperty( "Runmode", laserDriver_, "gam?", this, "gartn?", "sartn" );
    }
    
    property->SetCaching( false );

    property->RegisterEnumerationItem( "0", "ecc", EnumerationItem_RunMode_ConstantCurrent );
    property->RegisterEnumerationItem( "1", "ecp", EnumerationItem_RunMode_ConstantPower );

    RegisterPublicProperty( property );
}

void Gen5Laser::CreateFaultProperty()
{
    ImmutableEnumerationProperty* property = new ImmutableEnumerationProperty( "Laser Fault", laserDriver_, "f?" );

    property->RegisterEnumerationItem( "0", "No Fault" );
    property->RegisterEnumerationItem( "1", "TEC Fault" );
    property->RegisterEnumerationItem( "3", "Interlock" );
    property->RegisterEnumerationItem( "4", "Current Clamp" );

    property->SetCaching( false );

    RegisterPublicProperty( property );
}
