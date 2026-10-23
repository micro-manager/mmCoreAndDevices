/*
 * Micro-Manager device adapter for Hamilton devices that use the RNO protocol
 *
 * Author: Mark A. Tsuchida <mark@open-imaging.com> for the original MVP code
 *         Egor Zindy <ezindy@gmail.com> for the PSD additions
 *
 * Copyright (C) 2018 Applied Materials, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>
#include <vector>

#define RESOLUTION_HALF 0
#define RESOLUTION_FULL 1

enum DeviceType {
   DeviceTypeUnknown = 0,
   DeviceTypeMVP,
   DeviceTypePSD2,
   DeviceTypePSD3, //FIXME No firmware string
   DeviceTypePSD4, //FIXME No firmware string
   DeviceTypePSD6, //FIXME No firmware string
   DeviceTypePSD8, //FIXME No firmware string
   DeviceTypeML500A,
   DeviceTypeML500B,
   DeviceTypeML500C,
   DeviceTypeML600,
   DeviceTypeML700, //FIXME No firmware string
   DeviceTypeML900,
};

inline DeviceType GetDeviceTypeFromFirmware(std::string firmware)
{
   if (firmware.length() < 4)
      return DeviceTypeUnknown;

   // What I found:
   // AV07 = ML900, original
   // AV08 = ML900, PSD/2 valve
   // AV09 = ML900, ML900 valve
   // BV01 = ML500A, original
   // BV02 = ML500A, current
   // CV01 = ML500B and ML500C
   // DV01 = ML500C, OEM
   // NV01 = ML600, OEM
   // OM01 = PSD/2
   // MV = MVP

   if (firmware.rfind("MV",0) == 0)
      return DeviceTypeMVP;

   firmware = firmware.substr(0,4);

   if (firmware == "BV01" || firmware == "BV02")
      return DeviceTypeML500A; 
   if (firmware == "BV01" || firmware == "BV02")
      return DeviceTypeML500A; 
   else if (firmware == "CV01")
      return DeviceTypeML500B; 
   else if (firmware == "DV01")
      return DeviceTypeML500C; 
   else if (firmware == "NV01")
      return DeviceTypeML600; 
   else if (firmware == "OM01")
      return DeviceTypePSD2; 
   else
      return DeviceTypeUnknown;
}

inline std::string GetDeviceTypeName(DeviceType dt)
{
   switch (dt)
   {
      case DeviceTypeMVP:
         return "Modular Valve Positioner";
      case DeviceTypePSD2:
         return "PSD/2 Full-Height Syringe Pump";
      case DeviceTypePSD3:
         return "PSD/3 Half-Height Syringe Pump";
      case DeviceTypePSD4:
         return "PSD/4 Half-Height Syringe Pump";
      case DeviceTypePSD6:
         return "PSD/6 Full-Height Syringe Pump";
      case DeviceTypePSD8:
         return "PSD/8 Full-Height Syringe Pump";
      case DeviceTypeML500A:
         return "Microlab 500A";
      case DeviceTypeML500B:
         return "Microlab 500B";
      case DeviceTypeML500C:
         return "Microlab 500C";
      case DeviceTypeML600:
         return "Microlab 600";
      case DeviceTypeML700:
         return "Microlab 700";
      default:
         return "Unknown Device Device";
   }
}

inline long GetSyringeThrowSteps(DeviceType dt, int resolution)
{
   switch (dt)
   {
      case DeviceTypePSD3: //FIXME make sure we have the right values here
        return (resolution == RESOLUTION_HALF)?30000:30000;
      case DeviceTypePSD4: case DeviceTypePSD8:
        return (resolution == RESOLUTION_HALF)?3000:24000;
      case DeviceTypePSD6:
        return (resolution == RESOLUTION_HALF)?6000:48000;
      case DeviceTypeML600: //FIXME make sure we have the right values here
        return 52800;
      default:
        // PDS2, FIXME Check if not valid for ML500(A,B,C)
        return (resolution == RESOLUTION_HALF)?1000:2000;
   }
}
