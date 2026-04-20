///////////////////////////////////////////////////////////////////////////////
// FILE:       LegacyLaser.h
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

#ifndef __COBOLT__LEGACY_LASER_H
#define __COBOLT__LEGACY_LASER_H

#include <string>
#include <set>
#include <map>
#include <vector>

#include "Laser.h"

NAMESPACE_COBOLT_BEGIN

class LaserDriver;
class LaserStateProperty;
class LaserShutterProperty;
class MutableDeviceProperty;

class LegacyLaser : public Laser
{
public:

    typedef std::map<std::string, cobolt::Property*>::iterator PropertyIterator;

    LegacyLaser( const std::string& name, LaserDriver* driver );

    virtual ~LegacyLaser();

    const std::string& GetId() const;
    const std::string& GetName() const;

    void SetOn( const bool );
    void SetShutterOpen( const bool );

    virtual bool IsShutterEnabled() const;
    
    bool IsShutterOpen() const;

protected:

    static int NextId__;

    /// ###
    /// Property Generators
    
    virtual void CreateAutostartControlProperty() override;
    virtual void CreateClearFaultProperty() override;
    virtual void CreateNameProperty() override;
    virtual void CreateModelProperty() override;
    virtual void CreateWavelengthProperty( const std::string& wavelength ) override;
    virtual void CreateKeyswitchProperty() override;
    virtual void CreateSerialNumberProperty() override;
    virtual void CreateFirmwareVersionProperty() override;
    virtual void CreateAdapterVersionProperty() override;
     
    virtual void CreateOperatingHoursProperty() override;
    virtual void CreateCcCurrentSetpointProperty() override;
    virtual void CreateCcCurrentSetpointProperty( const std::string& getPersistedDataCommand, const std::string& setPersistedDataCommand );
    virtual void CreateCurrentReadingProperty() override;
    virtual void CreateCpPowerSetpointProperty() override;
    virtual void CreatePowerReadingProperty() override;
    
    virtual void CreateLaserOnOffProperty() override;
    virtual void CreateShutterProperty( std::string saveCmd = "sdsn", std::string readCmd = "gdsn?" );
    virtual void CreateCmDigitalModulationProperty() override;
    virtual void CreateCmAnalogModulationProperty() override;
     
    virtual void CreatePmPowerSetpointProperty() override;
    virtual void CreateAnalogImpedanceProperty() override;
     
    virtual void CreateModulationCurrentLowSetpointProperty();
    virtual void CreateModulationCurrentHighSetpointProperty();
    virtual void CreateModulationHighPowerSetpointProperty();

    virtual bool IsShutterCommandSupported() const;
    bool IsInCdrhMode() const;

    void RegisterPublicProperty( Property* );

    double MaxCurrentSetpoint();
    double MaxPowerSetpoint();
    
    LaserStateProperty* laserStatePropertyOld_;
};

NAMESPACE_COBOLT_END

#endif // #ifndef __COBOLT__LEGACY_LASER_H
