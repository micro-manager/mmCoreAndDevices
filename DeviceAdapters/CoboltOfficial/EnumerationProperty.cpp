///////////////////////////////////////////////////////////////////////////////
// FILE:       EnumerationProperty.cpp
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

#include "EnumerationProperty.h"

NAMESPACE_COBOLT_BEGIN

EnumerationProperty::EnumerationProperty( const std::string& name, LaserDriver* laserDriver, const std::string& commandName ) :
    MutableDeviceProperty( Property::Stereotype::String, name, laserDriver, commandName + "?" ),
    commandName_( commandName )
{
    std::string response;
    laserDriver_->SendCommand( commandName + "? options", &response );

    const char* cstr = response.c_str();
    do {
        if ( *cstr == ' ' ) {
            continue;
        }

        const char* begin = cstr;

        while ( *cstr != ',' && *cstr ) {
            cstr++;
        }

        enumerationItems_.push_back( std::string( begin, cstr ) );

    } while ( *cstr++ != 0 );
}

int EnumerationProperty::IntroduceToGuiEnvironment( GuiEnvironment* environment )
{
    for ( std::vector<std::string>::const_iterator enumerationItem = enumerationItems_.begin();
          enumerationItem != enumerationItems_.end();
          enumerationItem++ ) {

        const int returnCode = environment->RegisterAllowedGuiPropertyValue( GetName(), *enumerationItem );
        if ( returnCode != return_code::ok ) {
            return returnCode;
        }

        Logger::Instance()->LogMessage( "EnumerationProperty[ " + GetName() + " ]::IntroduceToGuiEnvironment(): Registered valid value '" +
            *enumerationItem + "' in GUI.", true );
    }

    return return_code::ok;
}

int EnumerationProperty::GetValue( std::string& string ) const
{
    Parent::GetValue( string );

    if ( string == "<undefined>" ) {

        SetToUnknownValue( string );
        Logger::Instance()->LogError( "EnumerationProperty[" + GetName() + "]::GetValue( ... ): Got '<undefined>' as a value" );
        return return_code::error; // Not 'invalid_value', as the cause is not the user.
    }

    return return_code::ok;
}

int EnumerationProperty::SetValue( const std::string& enumerationItemName )
{
    for ( std::vector<std::string>::const_iterator enumerationItem = enumerationItems_.begin();
          enumerationItem != enumerationItems_.end();
          enumerationItem++ ) {

        if ( enumerationItemName == *enumerationItem ) {
            return laserDriver_->SendCommand( commandName_ + " " + enumerationItemName );
        }
    }

    Logger::Instance()->LogError( "EnumerationProperty[ " + GetName() + " ]::SetValue(): Invalid enumeration item '" + enumerationItemName + "'" );
    return return_code::invalid_property_value;
}

bool EnumerationProperty::IsValidValue( const std::string& enumerationItemName )
{
    for ( std::vector<std::string>::const_iterator enumerationItem = enumerationItems_.begin();
        enumerationItem != enumerationItems_.end();
        enumerationItem++ ) {

        if ( enumerationItemName == *enumerationItem ) {
            return true;
        }
    }

    return false;
}

NAMESPACE_COBOLT_END
