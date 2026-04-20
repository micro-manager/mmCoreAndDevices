///////////////////////////////////////////////////////////////////////////////
// FILE:       Laser.h
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

#ifndef __COBOLT__LASER_H
#define __COBOLT__LASER_H

#include <string>
#include <set>
#include <map>
#include <vector>

#include "base.h"
#include "Property.h"

class StaticStringProperty;

NAMESPACE_COBOLT_BEGIN

class LaserDriver;
class DeviceProperty;
class LaserShutterProperty;
class MutableDeviceProperty;

class Laser
{
public:

    static const std::string Milliamperes;
    static const std::string Amperes;
    static const std::string Milliwatts;
    static const std::string Watts;

    static const std::string EnumerationItem_On;
    static const std::string EnumerationItem_Off;
    static const std::string EnumerationItem_Enabled;
    static const std::string EnumerationItem_Disabled;

    static const std::string EnumerationItem_RunMode_ConstantCurrent;
    static const std::string EnumerationItem_RunMode_ConstantPower;
    static const std::string EnumerationItem_RunMode_Modulation;

    typedef std::map<std::string, cobolt::Property*>::iterator PropertyIterator;

    ~Laser();

    const std::string& GetId() const;
    const std::string& GetName() const;

    void SetOn( const bool );
    void SetShutterOpen( const bool );

    virtual bool IsShutterEnabled() const;

    bool IsShutterOpen() const;

    Property* GetProperty( const std::string& name ) const;
    Property* GetProperty( const std::string& name );

    PropertyIterator GetPropertyIteratorBegin();
    PropertyIterator GetPropertyIteratorEnd();

protected:

    static int NextId__;

    Laser( const std::string& name, const std::string& wavelength, LaserDriver* driver );

    /// ###
    /// Property Generators
    virtual void CreateAdapterVersionProperty();
    virtual void CreateAnalogImpedanceProperty();
    virtual void CreateCmAnalogModulationProperty();
    virtual void CreatePmAnalogModulationProperty();
    virtual void CreateAutostartControlProperty();
    virtual void CreateCcCurrentSetpointProperty();
    virtual void CreateClearFaultProperty();
    virtual void CreateCmCurrentHighSetpointProperty();
    virtual void CreateCpPowerSetpointProperty();
    virtual void CreateCurrentReadingProperty();
    virtual void CreateCmDigitalModulationProperty();
    virtual void CreatePmDigitalModulationProperty();
    virtual void CreateFaultProperty();
    virtual void CreateFirmwareVersionProperty();
    virtual void CreateKeyswitchProperty();
    virtual void CreateLaserOnOffProperty();
    virtual void CreateLaserStateProperty();
    virtual void CreateModelProperty();
    virtual void CreateModulationInputVoltageMaxProperty();
    virtual void CreateNameProperty();
    virtual void CreateOperatingHoursProperty();
    virtual void CreatePmPowerSetpointProperty();
    virtual void CreatePowerReadingProperty();
    virtual void CreateRunmodeProperty();
    virtual void CreateSerialNumberProperty();
    virtual void CreateShutterProperty();
    virtual void CreateWavelengthProperty( const std::string& wavelength );

    virtual bool IsShutterCommandSupported() const;
    bool IsInCdrhMode() const;

    void RegisterPublicProperty( Property* );

    double MaxCurrentSetpoint();
    double MaxPowerSetpoint();

    std::map<std::string, cobolt::Property*> properties_;

    std::string id_;
    std::string name_;
    LaserDriver* laserDriver_;

    std::string currentUnit_;
    std::string powerUnit_;

    MutableDeviceProperty* laserOnOffProperty_;
    LaserShutterProperty* shutter_;

    DeviceProperty* laserStateProperty_;
};

NAMESPACE_COBOLT_END

#endif // #ifndef __COBOLT__LASER_H
