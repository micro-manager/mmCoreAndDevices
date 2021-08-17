/**
 * \file        Laser_TestSuite.h
 *
 * \authors     Lukas Kalinski
 *
 * \copyright   Cobolt AB, 2020. All rights reserved.
 */

#include <cxxtest/TestSuite.h>
#include "Laser.h"

using namespace cobolt;

class GuiPropertyMock : public GuiProperty
{
public:
    
    GuiPropertyMock( const std::string& string ) :
        value( string )
    {}

    virtual bool Set( const std::string& string ) { value = string; return true; }
    virtual bool Get( std::string& string ) const { string = value; return true; }
    std::string value;
};

class Laser_TestSuite : public CxxTest::TestSuite, public LaserDriver
{
    struct PhysicalLaserMock
    {
        PhysicalLaserMock() :
            firmwareVersion( "1.2.3" ),
            isOn( false ),
            isPauseCommandSupported( true )
        {}

        std::string firmwareVersion;
        bool isOn;
        bool isPauseCommandSupported;
    };

    PhysicalLaserMock _physicalLaserMock;
    Laser* _someLaser;
    std::string _lastReceivedCommand;

public:

    void setUp()
    {
        _someLaser = Laser::Create( this );
        _lastReceivedCommand.clear();
    }

    void tearDown()
    {
        delete _someLaser;
    }

    void test_GetProperty_firmware()
    {
        TS_ASSERT_EQUALS( _someLaser->GetProperty( "Firmware Version" )->GetValue(), _physicalLaserMock.firmwareVersion );
    }

    void test_OnGuiSetAction_toggle_on()
    {
        /// ###
        /// Setup

        _someLaser->SetOn( false );
        GuiPropertyMock guiProperty( "On" );
        
        /// ###
        /// Verify Setup

        TS_ASSERT( !_someLaser->IsOn() );
        
        /// ###
        /// Test

        _someLaser->GetProperty( "On-Off Switch" )->OnGuiSetAction( guiProperty );

        /// ###
        /// Verify
        
        TS_ASSERT( _physicalLaserMock.isOn );
    }

    void test_OnGuiSetAction_toggle_off()
    {
        /// ###
        /// Setup

        _physicalLaserMock.isOn = true;
        GuiPropertyMock guiProperty( "Off" );

        /// ###
        /// Test

        _someLaser->GetProperty( "On-Off Switch" )->OnGuiSetAction( guiProperty );

        /// ###
        /// Verify

        TS_ASSERT( !_physicalLaserMock.isOn );
    }

    /// ###
    /// Laser Device API

    virtual int SendCommand( const std::string& command, std::string* response = NULL )
    {
        if ( command == "gfv?" ) { *response = _physicalLaserMock.firmwareVersion; }
        if ( command == "l1" )   { _physicalLaserMock.isOn = true; }
        if ( command == "l0" )   { _physicalLaserMock.isOn = false; }
        if ( command == "glm?" ) { *response = "0485-06-01"; }
        
        _lastReceivedCommand = command;

        return cobolt::return_code::ok;
    }
};