// Micro-Manager Device Adapter for Backer & Hickl DCC/DCU
// Author: Mark A. Tsuchida
//
// Copyright 2023 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#pragma once

// Wrap the BH API for DCC and DCU into a (compile-time) common interface.

#include <dcc_def.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <string>
#include <utility>

template <typename ModInfo>
inline auto GetModInfoSerialNo(ModInfo const& mod_info) noexcept
    -> std::string {
   static constexpr std::size_t serial_no_siz = sizeof(mod_info.serial_no);
   std::array<char, serial_no_siz + 1> serial_no;
   serial_no.back() = '\0'; // Just in case
   std::copy(mod_info.serial_no, mod_info.serial_no + serial_no_siz,
             serial_no.begin());
   return serial_no.data();
}

// Drop-in wrappers for DCCModInfo and DCCUSBModInfo that provide a uniform
// accessors.

struct DCCModInfoWrap : DCCModInfo {
   auto ModuleType() const noexcept -> short { return module_type; }
   auto BusNumber() const noexcept -> short { return bus_number; }
   auto SlotNumber() const noexcept -> short { return slot_number; }
   auto InUse() const noexcept -> short { return in_use; }
   auto Init() const noexcept -> short { return init; }
   auto BaseAdr() const noexcept -> unsigned short { return base_adr; }
   auto ComPortNo() const noexcept -> short { return 0; }
   auto SerialNo() const noexcept -> std::string {
      return GetModInfoSerialNo(*this);
   }
};

struct DCCUSBModInfoWrap : DCCUSBModInfo {
   auto ModuleType() const noexcept -> short { return module_type; }
   auto BusNumber() const noexcept -> short { return 0; }
   auto SlotNumber() const noexcept -> short { return 0; }
   auto InUse() const noexcept -> short { return in_use; }
   auto Init() const noexcept -> short { return init; }
   auto BaseAdr() const noexcept -> unsigned short { return 0; }
   auto ComPortNo() const noexcept -> short { return com_port_no; }
   auto SerialNo() const noexcept -> std::string {
      return GetModInfoSerialNo(*this);
   }
};

enum class DCCOrDCU {
   DCC, // PCI(e)
   DCU, // USB
};

// Each connector of DCC or DCU has some subset of these features
enum class ConnectorFeature {
   Plus5V,
   Minus5V,
   Plus12V,
   GainHV,
   DigitalOut,
   Cooling,
   CoolerVoltage,
   CoolerCurrentLimit,

   // The above values correspond to the "parameters" in the BH API.
   // Additional features:
   // get_gain_HV_limit iff connector has GainHV
   // enable_outputs available for all connectors (individually iff DCU)
   // clear_overload iff connector has GainHV (individually iff DCU)
   // get_overload_state (bit in result) iff connector has GainHV
   // get_curr_lmt_state iff connector has CoolerCurrentLimit
};

// This function is shared between DCC and DCU
inline auto DCCDCUGetErrorString(short err) noexcept -> std::string {
   const std::string codeSuffix = " (" + std::to_string(err) + ")";
   std::array<char, 1024> buf;
   auto e =
       DCC_get_error_string(err, buf.data(), static_cast<short>(buf.size()));
   if (e) {
      return "Unknown DCC/DCU error" + codeSuffix;
   }
   return buf.data() + codeSuffix;
}

template <DCCOrDCU Model> class DCCDCUConfig;

template <> class DCCDCUConfig<DCCOrDCU::DCC> {
 public:
   static constexpr std::size_t MaxNrModules = 8;
   static constexpr std::size_t NrConnectors = 3;
   static constexpr short SimulationMode = DCC_SIMUL100;
   static constexpr bool HasIndividualOutputAndClearOverload = false;
   using ModInfoType = DCCModInfoWrap;

   static constexpr auto IniHeaderComment = "DCC100";
   static constexpr auto IniBaseTag = "dcc_base";
   static constexpr auto IniSimulationKey = "simulation";
   static constexpr auto IniModuleTagPrefix = "dcc_module";

   static auto Init(char const* ini_file) noexcept -> short {
      return DCC_init(const_cast<char*>(ini_file));
   }

   static auto TestIfActive(short mod_no) noexcept -> short {
      return DCC_test_if_active(mod_no);
   }

   static auto GetInitStatus(short mod_no, short* ini_status) noexcept
       -> short {
      return DCC_get_init_status(mod_no, ini_status);
   }

   static auto GetMode() noexcept -> short { return DCC_get_mode(); }

   static auto GetModuleInfo(short mod_no, ModInfoType* mod_info) noexcept
       -> short {
      return DCC_get_module_info(mod_no, mod_info);
   }

   static auto GetParameter(short mod_no, short par_id, float* value) noexcept
       -> short {
      return DCC_get_parameter(mod_no, par_id, value);
   }

   static auto SetParameter(short mod_no, short par_id, short send_to_hard,
                            float value) noexcept -> short {
      return DCC_set_parameter(mod_no, par_id, send_to_hard, value);
   }

   static auto GetGainHVLimit(short mod_no, short con_no,
                              short* value) noexcept -> short {
      assert(con_no == 0 || con_no == 2);
      return DCC_get_gain_HV_limit(mod_no, con_no == 0 ? 0 : 1, value);
   }

   static auto EnableAllOutputs(short mod_no, short enable) noexcept -> short {
      return DCC_enable_outputs(mod_no, enable);
   }

   static auto EnableOutput(short mod_no, short con_no, short enable) noexcept
       -> short {
      (void)mod_no;
      (void)con_no;
      (void)enable;
      assert(false);
      std::terminate();
   }

   static auto ClearAllOverloads(short mod_no) noexcept -> short {
      return DCC_clear_overload(mod_no);
   }

   static auto ClearOverload(short mod_no, short con_no) noexcept -> short {
      (void)mod_no;
      (void)con_no;
      assert(false);
      std::terminate();
   }

   static auto GetOverloadState(short mod_no, short* state) noexcept -> short {
      // Bit 0 = Connector 1; Bit 1 = Connector 3:
      short const ret = DCC_get_overload_state(mod_no, state);
      // Make consistent with DCU (bit no  = connector no - 1):
      short const conn1bit = *state & 1;
      *state >>= 1;
      *state <<= 2;
      *state |= conn1bit;
      return ret;
   }

   static auto GetCurrLmtState(short mod_no, short* state) noexcept -> short {
      // Bit 0 = Connector 3:
      short const ret = DCC_get_curr_lmt_state(mod_no, state);
      // Make consistent with DCU (bit 2 = connector 3):
      *state <<= 2;
      return ret;
   }

   static auto Close() noexcept -> short { return DCC_close(); }

   static auto ConnectorHasFeature(short con_no,
                                   ConnectorFeature feature) noexcept -> bool {
      switch (con_no) {
      case 0:
         return feature == ConnectorFeature::Plus5V ||
                feature == ConnectorFeature::Minus5V ||
                feature == ConnectorFeature::Plus12V ||
                feature == ConnectorFeature::GainHV;
      case 1:
         return feature == ConnectorFeature::Plus5V ||
                feature == ConnectorFeature::Minus5V ||
                feature == ConnectorFeature::Plus12V ||
                feature == ConnectorFeature::DigitalOut;
      case 2:
         return feature == ConnectorFeature::Plus5V ||
                feature == ConnectorFeature::Minus5V ||
                feature == ConnectorFeature::Plus12V ||
                feature == ConnectorFeature::GainHV ||
                feature == ConnectorFeature::Cooling ||
                feature == ConnectorFeature::CoolerVoltage ||
                feature == ConnectorFeature::CoolerCurrentLimit;
      default:
         return false;
      }
   }

   static auto ConnectorParameterId(short con_no,
                                    ConnectorFeature feature) noexcept
       -> short {
      auto const p = std::make_pair(static_cast<int>(con_no), feature);

      if (p == std::make_pair(0, ConnectorFeature::Plus5V))
         return C1_P5V;
      if (p == std::make_pair(0, ConnectorFeature::Minus5V))
         return C1_M5V;
      if (p == std::make_pair(0, ConnectorFeature::Plus12V))
         return C1_P12V;
      if (p == std::make_pair(0, ConnectorFeature::GainHV))
         return C1_GAIN_HV;

      if (p == std::make_pair(1, ConnectorFeature::Plus5V))
         return C2_P5V;
      if (p == std::make_pair(1, ConnectorFeature::Minus5V))
         return C2_M5V;
      if (p == std::make_pair(1, ConnectorFeature::Plus12V))
         return C2_P12V;
      if (p == std::make_pair(1, ConnectorFeature::DigitalOut))
         return C2_DIGOUT;

      if (p == std::make_pair(2, ConnectorFeature::Plus5V))
         return C3_P5V;
      if (p == std::make_pair(2, ConnectorFeature::Minus5V))
         return C3_M5V;
      if (p == std::make_pair(2, ConnectorFeature::Plus12V))
         return C3_P12V;
      if (p == std::make_pair(2, ConnectorFeature::GainHV))
         return C3_GAIN_HV;
      if (p == std::make_pair(2, ConnectorFeature::Cooling))
         return C3_COOLING;
      if (p == std::make_pair(2, ConnectorFeature::CoolerVoltage))
         return C3_COOLVOLT;
      if (p == std::make_pair(2, ConnectorFeature::CoolerCurrentLimit))
         return C3_COOLCURR;

      assert(false);
      std::terminate();
   }
};

template <> class DCCDCUConfig<DCCOrDCU::DCU> {
 public:
   static constexpr std::size_t MaxNrModules = 4;
   static constexpr std::size_t NrConnectors = 5;
   static constexpr short SimulationMode = DCC_SIMULUSB;
   static constexpr bool HasIndividualOutputAndClearOverload = true;
   using ModInfoType = DCCUSBModInfoWrap;

   static constexpr auto IniHeaderComment = "DCCUSB";
   static constexpr auto IniBaseTag = "dccusb_base";
   static constexpr auto IniSimulationKey = "simulation_usb";
   static constexpr auto IniModuleTagPrefix = "dccusb_module";

   static auto Init(char const* ini_file) noexcept -> short {
      return DCCUSB_init(const_cast<char*>(ini_file));
   }

   static auto TestIfActive(short mod_no) noexcept -> short {
      return DCCUSB_test_if_active(mod_no);
   }

   static auto GetInitStatus(short mod_no, short* ini_status) noexcept
       -> short {
      return DCCUSB_get_init_status(mod_no, ini_status);
   }

   static auto GetMode() noexcept -> short { return DCCUSB_get_mode(); }

   static auto GetModuleInfo(short mod_no, ModInfoType* mod_info) noexcept
       -> short {
      return DCCUSB_get_module_info(mod_no, mod_info);
   }

   static auto GetParameter(short mod_no, short par_id, float* value) noexcept
       -> short {
      return DCCUSB_get_parameter(mod_no, par_id, value);
   }

   static auto SetParameter(short mod_no, short par_id, short send_to_hard,
                            float value) noexcept -> short {
      return DCCUSB_set_parameter(mod_no, par_id, send_to_hard, value);
   }

   static auto GetGainHVLimit(short mod_no, short con_no,
                              short* value) noexcept -> short {
      assert(0 <= con_no && con_no < 4);
      return DCCUSB_get_gain_HV_limit(mod_no, con_no, value);
   }

   static auto EnableAllOutputs(short mod_no, short enable) noexcept -> short {
      return DCCUSB_enable_outputs(mod_no, -1, enable);
   }

   static auto EnableOutput(short mod_no, short con_no, short enable) noexcept
       -> short {
      assert(0 <= con_no && con_no < 4);
      return DCCUSB_enable_outputs(mod_no, con_no, enable);
   }

   static auto ClearAllOverloads(short mod_no) noexcept -> short {
      return DCCUSB_clear_overload(mod_no, -1);
   }

   static auto ClearOverload(short mod_no, short con_no) noexcept -> short {
      assert(0 <= con_no && con_no < 3);
      return DCCUSB_clear_overload(mod_no, con_no);
   }

   static auto GetOverloadState(short mod_no, short* state) noexcept -> short {
      return DCCUSB_get_overload_state(mod_no, state);
   }

   static auto GetCurrLmtState(short mod_no, short* state) noexcept -> short {
      return DCCUSB_get_curr_lmt_state(mod_no, state);
   }

   static auto Close() noexcept -> short { return DCCUSB_close(); }

   static auto ConnectorHasFeature(short con_no,
                                   ConnectorFeature feature) noexcept -> bool {
      if (0 <= con_no && con_no < 4) {
         return feature == ConnectorFeature::Plus5V ||
                feature == ConnectorFeature::Minus5V ||
                feature == ConnectorFeature::Plus12V ||
                feature == ConnectorFeature::GainHV ||
                feature == ConnectorFeature::Cooling ||
                feature == ConnectorFeature::CoolerVoltage ||
                feature == ConnectorFeature::CoolerCurrentLimit;
      }
      if (con_no == 4) {
         return feature == ConnectorFeature::DigitalOut;
      }
      return false;
   }

   static auto ConnectorParameterId(short con_no,
                                    ConnectorFeature feature) noexcept
       -> short {
      auto const p = std::make_pair(static_cast<int>(con_no), feature);

      if (p == std::make_pair(0, ConnectorFeature::Plus5V))
         return C1U_P5V;
      if (p == std::make_pair(0, ConnectorFeature::Minus5V))
         return C1U_M5V;
      if (p == std::make_pair(0, ConnectorFeature::Plus12V))
         return C1U_P12V;
      if (p == std::make_pair(0, ConnectorFeature::GainHV))
         return C1U_GAIN_HV;
      if (p == std::make_pair(0, ConnectorFeature::Cooling))
         return C1U_COOLING;
      if (p == std::make_pair(0, ConnectorFeature::CoolerVoltage))
         return C1U_COOLVOLT;
      if (p == std::make_pair(0, ConnectorFeature::CoolerCurrentLimit))
         return C1U_COOLCURR;

      if (p == std::make_pair(1, ConnectorFeature::Plus5V))
         return C2U_P5V;
      if (p == std::make_pair(1, ConnectorFeature::Minus5V))
         return C2U_M5V;
      if (p == std::make_pair(1, ConnectorFeature::Plus12V))
         return C2U_P12V;
      if (p == std::make_pair(1, ConnectorFeature::GainHV))
         return C2U_GAIN_HV;
      if (p == std::make_pair(1, ConnectorFeature::Cooling))
         return C2U_COOLING;
      if (p == std::make_pair(1, ConnectorFeature::CoolerVoltage))
         return C2U_COOLVOLT;
      if (p == std::make_pair(1, ConnectorFeature::CoolerCurrentLimit))
         return C2U_COOLCURR;

      if (p == std::make_pair(2, ConnectorFeature::Plus5V))
         return C3U_P5V;
      if (p == std::make_pair(2, ConnectorFeature::Minus5V))
         return C3U_M5V;
      if (p == std::make_pair(2, ConnectorFeature::Plus12V))
         return C3U_P12V;
      if (p == std::make_pair(2, ConnectorFeature::GainHV))
         return C3U_GAIN_HV;
      if (p == std::make_pair(2, ConnectorFeature::Cooling))
         return C3U_COOLING;
      if (p == std::make_pair(2, ConnectorFeature::CoolerVoltage))
         return C3U_COOLVOLT;
      if (p == std::make_pair(2, ConnectorFeature::CoolerCurrentLimit))
         return C3U_COOLCURR;

      if (p == std::make_pair(3, ConnectorFeature::Plus5V))
         return C4U_P5V;
      if (p == std::make_pair(3, ConnectorFeature::Minus5V))
         return C4U_M5V;
      if (p == std::make_pair(3, ConnectorFeature::Plus12V))
         return C4U_P12V;
      if (p == std::make_pair(3, ConnectorFeature::GainHV))
         return C4U_GAIN_HV;
      if (p == std::make_pair(3, ConnectorFeature::Cooling))
         return C4U_COOLING;
      if (p == std::make_pair(3, ConnectorFeature::CoolerVoltage))
         return C4U_COOLVOLT;
      if (p == std::make_pair(3, ConnectorFeature::CoolerCurrentLimit))
         return C4U_COOLCURR;

      if (p == std::make_pair(4, ConnectorFeature::DigitalOut))
         return C5U_DIGOUT;

      assert(false);
      std::terminate();
   }
};
