# CMakeLists.txt in mmCoreAndDevices/MMDevice/DeviceAdapters

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.15)

# Include necessary modules
include(CheckIncludeFiles)
include(CheckLibraryExists)

# Project configuration
project(DeviceAdapters)

# Check for C++14 support
set(CMAKE_CXX_STANDARD 14)

# Set the directory containing additional CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../m4")

# Include necessary libraries
#include(MMdevAPI)
#include(MMgmock)
#include(MMlibusb0_1)
#include(MMhidapi)
#include(MMlibmodbus)
#include(MMlibdc1394)
#include(MMlibgphoto2)
#include(MMFreeImagePlus)
#include(MMlibmsgpack)
#include(MMlibopencv)

# Define the sources for device adapter libraries
set(device_adapter_dirs
    # AAAOTF
    # AOTF
    # ASIFW1000
    # ASIStage
    # ASITiger
    # ASIWPTR
    # Aladdin
    # Andor
    # AndorLaserCombiner
    # AndorSDK3
    # Aquinas
    # Arduino
    # Arduino32bitBoards
    # Basler
    # BlueboxOptics_niji
    # CARVII
    # CNCMicroscope
    # CNCMicroscope/ArduinoNeoPixelShutter
    # CNCMicroscope/RAMPSStage
    # CSUW1
    # ChuoSeiki_MD5000
    # ChuoSeiki_QT
    # Cobolt
    # CoherentCube
    # CoherentOBIS
    # CoherentScientificRemote
    # Conix
    # CoolLEDpE300
    # CoolLEDpE4000
    # Corvus
    # DTOpenLayer
    DemoCamera
    # Diskovery
    # FakeCamera
    # FocalPoint
    # FreeSerialPort
    # HIDManager
    # HamiltonMVP
    # HamiltonMVP/unittest
    # HydraLMT200
    # IDS_uEye
    # IIDC
    # ITC18
    # ImageProcessorChain
    # IsmatecMCP
    # K8055
    # K8061
    # LaserQuantumLaser
    # LeicaDMI
    # LeicaDMR
    # LeicaDMSTC
    # LightSheetManager
    # Ludl
    # LudlLow
    # LumencorSpectra
    # MP285
    # MPBLaser
    # MaestroServo
    # Marzhauser
    # Marzhauser-LStep
    # MarzhauserLStepOld
    # MicroFPGA
    # MicroPoint
    # Motic_mac
    # Neos
    # NewportCONEX
    # NewportSMC
    # Nikon
    # NikonTE2000
    # OVP_ECS2
    # Omicron
    # OpenCVgrabber
    # Oxxius
    # OxxiusCombiner
    # PI
    # PIEZOCONCEPT
    # PI_GCS
    # PI_GCS_2
    # PVCAM
    # ParallelPort
    # Pecon
    # Piezosystem_30DV50
    # Piezosystem_NV120_1
    # Piezosystem_NV40_1
    # Piezosystem_NV40_3
    # Piezosystem_dDrive
    # PrecisExcite
    # Prior
    # PriorLegacy
    # QCam
    # Sapphire
    # Scientifica
    # ScionCam
    # Sensicam
    # SequenceTester
    # SerialManager
    # SimpleCam
    # Skyra
    # SmarActHCU-3D
    # SouthPort
    # SpectralLMM5
    # Spot
    # StarlightXpress
    # SutterLambda
    # SutterLambda2
    # SutterLambdaParallelArduino
    # SutterStage
    # Thorlabs
    # ThorlabsDCxxxx
    # ThorlabsElliptecSlider
    # ThorlabsFilterWheel
    # ThorlabsSC10
    # Thorlabs_ELL14
    # Tofra
    # Toptica_iBeamSmartCW
    # TriggerScope
    # TriggerScopeMM
    # USBManager
    # UserDefinedSerial
    # UserDefinedSerial/unittest
    # Utilities
    # VariLC
    # VarispecLCTF
    # Video4Linux
    # Vincent
    # Vortran
    # WieneckeSinske
    # XCite120PC_Exacte
    # XCiteLed
    # XCiteXT600
    # XLight
    # XLightV3
    # Xcite
    # YodnE600
    # Yokogawa
    # Zaber
    # ZeissCAN
    # ZeissCAN29
    # dc1394
    # kdv
    # nPoint
    # pgFocus
)

# Create device adapter libraries
foreach(dir ${device_adapter_dirs})
    add_subdirectory(${dir})
endforeach()
