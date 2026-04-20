///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentLensDatabase.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident objective lens database from SDK LensInfo.unit
//
// COPYRIGHT:     University of California, San Francisco, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include <string>
#include <vector>

namespace EvidentLens {

// Immersion medium types (from SDK SubType values)
enum ImmersionType
{
   Immersion_Dry = 0x00106010,
   Immersion_Water = 0x00106011,
   Immersion_Oil = 0x00106012,
   Immersion_Silicon = 0x00106013,
   Immersion_Gel = 0x00106016
};

// Convert immersion type to string
inline const char* ImmersionTypeToString(ImmersionType type)
{
   switch (type)
   {
      case Immersion_Dry: return "Dry";
      case Immersion_Water: return "Water";
      case Immersion_Oil: return "Oil";
      case Immersion_Silicon: return "Silicon";
      case Immersion_Gel: return "Gel";
      default: return "Unknown";
   }
}

// Lens information structure
struct LensInfo
{
   const char* model;
   unsigned int modelCode;
   double magnification;
   ImmersionType immersion;
   double na;
   double wd;
};

// Complete lens database from SDK LensInfo.unit file
static const LensInfo LENS_DATABASE[] = {
   // Lens0
   {"UPLSAPO4X", 0x125010, 4.0, Immersion_Dry, 0.160, 13.0},
   // Lens1
   {"UPLSAPO10X2", 0x125011, 10.0, Immersion_Dry, 0.400, 3.1},
   // Lens2
   {"UPLSAPO20X", 0x125012, 20.0, Immersion_Dry, 0.750, 0.6},
   // Lens3
   {"UPLSAPO20XO", 0x125013, 20.0, Immersion_Oil, 0.850, 0.17},
   // Lens4
   {"UPLSAPO30XS", 0x125014, 30.0, Immersion_Silicon, 1.050, 0.8},
   // Lens5
   {"UPLSAPO40X2", 0x125015, 40.0, Immersion_Dry, 0.950, 0.18},
   // Lens6
   {"UPLSAPO60XW", 0x125016, 60.0, Immersion_Water, 1.200, 0.28},
   // Lens7
   {"UPLSAPO60XO", 0x125017, 60.0, Immersion_Oil, 1.350, 0.15},
   // Lens8
   {"UPLSAPO60XS", 0x125018, 60.0, Immersion_Silicon, 1.300, 0.3},
   // Lens9
   {"UPLSAPO100XO", 0x125019, 100.0, Immersion_Oil, 1.400, 0.13},
   // Lens10
   {"PLAPON1.25X", 0x12501a, 1.25, Immersion_Dry, 0.040, 5.0},
   // Lens11
   {"PLAPON2X", 0x12501b, 2.0, Immersion_Dry, 0.080, 6.2},
   // Lens12
   {"PLAPON60XO", 0x12501c, 60.0, Immersion_Oil, 1.420, 0.15},
   // Lens13
   {"PLAPON60XOSC", 0x12501d, 60.0, Immersion_Oil, 1.400, 0.12},
   // Lens14
   {"UPLFLN4X", 0x12501e, 4.0, Immersion_Dry, 0.130, 17.0},
   // Lens15
   {"UPLFLN10X2", 0x12501f, 10.0, Immersion_Dry, 0.300, 10.0},
   // Lens16
   {"UPLFLN20X", 0x125020, 20.0, Immersion_Dry, 0.500, 2.1},
   // Lens17
   {"UPLFLN40X", 0x125021, 40.0, Immersion_Dry, 0.750, 0.51},
   // Lens18
   {"UPLFLN40XO", 0x125022, 40.0, Immersion_Oil, 1.300, 0.2},
   // Lens19
   {"UPLFLN60X", 0x125023, 60.0, Immersion_Dry, 0.900, 0.2},
   // Lens20
   {"UPLFLN60XOI", 0x125024, 60.0, Immersion_Oil, 1.250, 0.12},
   // Lens21
   {"UPLFLN100XO2", 0x125025, 100.0, Immersion_Oil, 1.300, 0.2},
   // Lens22
   {"UPLFLN100XOI2", 0x125026, 100.0, Immersion_Oil, 1.300, 0.2},
   // Lens23
   {"UPLFLN4XPH", 0x125027, 4.0, Immersion_Dry, 0.130, 17.0},
   // Lens24
   {"UPLFLN10X2PH", 0x125028, 10.0, Immersion_Dry, 0.300, 10.0},
   // Lens25
   {"UPLFLN20XPH", 0x125029, 20.0, Immersion_Dry, 0.500, 2.1},
   // Lens26
   {"UPLFLN40XPH", 0x12502a, 40.0, Immersion_Dry, 0.750, 0.51},
   // Lens27
   {"UPLFLN60XOIPH", 0x12502b, 60.0, Immersion_Oil, 1.250, 0.12},
   // Lens28
   {"UPLFLN100XO2PH", 0x12502c, 100.0, Immersion_Oil, 1.300, 0.2},
   // Lens29
   {"UPLFLN4XP", 0x12502d, 4.0, Immersion_Dry, 0.130, 17.0},
   // Lens30
   {"UPLFLN10XP", 0x12502e, 10.0, Immersion_Dry, 0.300, 10.0},
   // Lens31
   {"UPLFLN20XP", 0x12502f, 20.0, Immersion_Dry, 0.500, 2.1},
   // Lens32
   {"UPLFLN40XP", 0x125030, 40.0, Immersion_Dry, 0.750, 0.51},
   // Lens33
   {"UPLFLN100XOP", 0x125031, 100.0, Immersion_Oil, 1.300, 0.2},
   // Lens34
   {"LUCPLFLN20X", 0x125032, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens35
   {"LUCPLFLN40X", 0x125033, 40.0, Immersion_Dry, 0.600, 2.7},
   // Lens36
   {"LUCPLFLN60X", 0x125034, 60.0, Immersion_Dry, 0.700, 1.5},
   // Lens37
   {"CPLFLN10XPH", 0x125035, 10.0, Immersion_Dry, 0.300, 9.5},
   // Lens38
   {"LUCPLFLN20XPH", 0x125036, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens39
   {"LUCPLFLN40XPH", 0x125037, 40.0, Immersion_Dry, 0.600, 3.0},
   // Lens40
   {"LUCPLFLN60XPH", 0x125038, 60.0, Immersion_Dry, 0.700, 1.5},
   // Lens41
   {"CPLFLN10XRC", 0x125039, 10.0, Immersion_Dry, 0.300, 9.0},
   // Lens42
   {"LUCPLFLN20XRC", 0x12503a, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens43
   {"LUCPLFLN40XRC", 0x12503b, 40.0, Immersion_Dry, 0.600, 3.0},
   // Lens44
   {"CPLN10XPH", 0x12503c, 10.0, Immersion_Dry, 0.250, 10.0},
   // Lens45
   {"LCACHN20XPH", 0x12503d, 20.0, Immersion_Dry, 0.400, 3.2},
   // Lens46
   {"LCACHN40XPH", 0x12503e, 40.0, Immersion_Dry, 0.550, 2.2},
   // Lens47
   {"CPLN10XRC", 0x12503f, 10.0, Immersion_Dry, 0.250, 9.7},
   // Lens48
   {"LCACHN20XRC", 0x125040, 20.0, Immersion_Dry, 0.400, 2.8},
   // Lens49
   {"LCACHN40XRC", 0x125041, 40.0, Immersion_Dry, 0.550, 1.9},
   // Lens50
   {"UAPON20XW340", 0x125042, 20.0, Immersion_Water, 0.700, 0.35},
   // Lens51
   {"UAPON40XO340", 0x125043, 40.0, Immersion_Oil, 1.350, 0.1},
   // Lens52
   {"UAPON40XW340", 0x125044, 40.0, Immersion_Water, 1.150, 0.25},
   // Lens53
   {"APON60XOTIRF", 0x125045, 60.0, Immersion_Oil, 1.490, 0.1},
   // Lens54
   {"UAPON100XOTIRF", 0x125046, 100.0, Immersion_Oil, 1.490, 0.1},
   // Lens55
   {"UAPON150XOTIRF", 0x125047, 150.0, Immersion_Oil, 1.450, 0.08},
   // Lens56
   {"MPLFLN1.25X", 0x125048, 1.25, Immersion_Dry, 0.040, 3.5},
   // Lens57
   {"MPLN5X", 0x125049, 5.0, Immersion_Dry, 0.100, 20.0},
   // Lens58
   {"LCPLFLN100XLCD", 0x12504a, 100.0, Immersion_Dry, 0.850, 0.9},
   // Lens59
   {"UCPLFLN20X", 0x12504b, 20.0, Immersion_Dry, 0.700, 0.8},
   // Lens60
   {"UCPLFLN20XPH", 0x12504c, 20.0, Immersion_Dry, 0.700, 0.8},
   // Lens61
   {"UPLSAPO100XOPH", 0x12504d, 100.0, Immersion_Oil, 1.400, 0.13},
   // Lens62
   {"APON100XHOTIRF", 0x12504e, 100.0, Immersion_Oil, 1.700, 0.08},
   // Lens63
   {"PLAPON60XOPH", 0x12504f, 60.0, Immersion_Oil, 1.420, 0.15},
   // Lens64
   {"UPLSAPO40XS", 0x125050, 40.0, Immersion_Silicon, 1.250, 0.3},
   // Lens65
   {"UAPON40XO340-2", 0x125051, 40.0, Immersion_Oil, 1.350, 0.1},
   // Lens66
   {"PLAPON60XOSC2", 0x125052, 60.0, Immersion_Oil, 1.400, 0.12},
   // Lens67
   {"LMPLFLN5X", 0x125053, 5.0, Immersion_Dry, 0.130, 22.5},
   // Lens68
   {"LMPLFLN10X", 0x125054, 10.0, Immersion_Dry, 0.250, 21.0},
   // Lens69
   {"LMPLFLN20X", 0x125055, 20.0, Immersion_Dry, 0.400, 12.0},
   // Lens70
   {"LMPLFLN50X", 0x125056, 50.0, Immersion_Dry, 0.500, 10.6},
   // Lens71
   {"UPLSAPO30XSIR", 0x125057, 30.0, Immersion_Silicon, 1.050, 0.8},
   // Lens72
   {"UPLSAPO100XS", 0x125058, 100.0, Immersion_Silicon, 1.350, 0.2},
   // Lens73
   {"LUMFLN60XW", 0x125059, 60.0, Immersion_Water, 1.100, 1.5},
   // Lens74
   {"UPLSAPO60XS2", 0x12505a, 60.0, Immersion_Silicon, 1.300, 0.3},
   // Lens75
   {"UPLXAPO4X", 0x12505b, 4.0, Immersion_Dry, 0.160, 13.0},
   // Lens76
   {"UPLXAPO10X", 0x12505c, 10.0, Immersion_Dry, 0.400, 3.1},
   // Lens77
   {"UPLXAPO20X", 0x12505d, 20.0, Immersion_Dry, 0.800, 0.6},
   // Lens78
   {"UPLXAPO40X", 0x12505e, 40.0, Immersion_Dry, 0.950, 0.18},
   // Lens79
   {"UPLXAPO40XO", 0x12505f, 40.0, Immersion_Oil, 1.400, 0.13},
   // Lens80
   {"UPLXAPO60XO", 0x125060, 60.0, Immersion_Oil, 1.420, 0.15},
   // Lens81
   {"UPLXAPO60XOPH", 0x125061, 60.0, Immersion_Oil, 1.420, 0.15},
   // Lens82
   {"UPLXAPO100XO", 0x125062, 100.0, Immersion_Oil, 1.450, 0.13},
   // Lens83
   {"UPLXAPO100XOPH", 0x125063, 100.0, Immersion_Oil, 1.450, 0.13},
   // Lens84
   {"UPLAPO60XOHR", 0x125064, 60.0, Immersion_Oil, 1.500, 0.11},
   // Lens85
   {"UPLAPO100XOHR", 0x125065, 100.0, Immersion_Oil, 1.500, 0.12},
   // Lens86
   {"LUCPLFLN20X2", 0x125066, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens87
   {"LUCPLFLN20XPH2", 0x125067, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens88
   {"LUCPLFLN20XRC2", 0x125068, 20.0, Immersion_Dry, 0.450, 6.6},
   // Lens89
   {"LUCPLFLN40X2", 0x125069, 40.0, Immersion_Dry, 0.600, 2.7},
   // Lens90
   {"LUCPLFLN40XPH2", 0x12506a, 40.0, Immersion_Dry, 0.600, 3.0},
   // Lens91
   {"LUCPLFLN40XRC2", 0x12506b, 40.0, Immersion_Dry, 0.600, 3.0},
   // Lens92
   {"LUCPLFLN60X2", 0x12506c, 60.0, Immersion_Dry, 0.700, 1.5},
   // Lens93
   {"LUCPLFLN60XPH2", 0x12506d, 60.0, Immersion_Dry, 0.700, 1.5},
   // Lens94
   {"UCPLFLN20X2", 0x12506e, 20.0, Immersion_Dry, 0.700, 0.8},
   // Lens95
   {"UCPLFLN20XPH2", 0x12506f, 20.0, Immersion_Dry, 0.700, 0.8},
   // Lens96
   {"UPLXAPO60XW", 0x125070, 60.0, Immersion_Water, 1.200, 0.28},
   // Lens97
   {"LUPLAPO25XS", 0x125071, 25.0, Immersion_Silicon, 0.850, 2.0},
   // Lens98
   {"LUPLAPO25XS_W", 0x125072, 25.0, Immersion_Water, 0.850, 2.0},
   // Lens99
   {"LUPLAPO25XS_GEL", 0x125073, 25.0, Immersion_Gel, 0.850, 2.0},
   // Lens100
   {"MPLFLN2.5X2", 0x125074, 2.5, Immersion_Dry, 0.080, 10.7},
   // Lens101
   {"MPLFLN5X2", 0x125075, 5.0, Immersion_Dry, 0.150, 20.0}
};

static const int LENS_DATABASE_SIZE = sizeof(LENS_DATABASE) / sizeof(LensInfo);

// Helper function: Get all lenses matching a specific magnification
inline std::vector<const LensInfo*> GetLensesByMagnification(double magnification)
{
   std::vector<const LensInfo*> result;
   for (int i = 0; i < LENS_DATABASE_SIZE; i++)
   {
      if (LENS_DATABASE[i].magnification == magnification)
      {
         result.push_back(&LENS_DATABASE[i]);
      }
   }
   return result;
}

// Helper function: Get all lenses matching a specific immersion type
inline std::vector<const LensInfo*> GetLensesByImmersion(ImmersionType immersion)
{
   std::vector<const LensInfo*> result;
   for (int i = 0; i < LENS_DATABASE_SIZE; i++)
   {
      if (LENS_DATABASE[i].immersion == immersion)
      {
         result.push_back(&LENS_DATABASE[i]);
      }
   }
   return result;
}

// Helper function: Get all lenses matching both magnification and immersion type
inline std::vector<const LensInfo*> GetLensesByMagnificationAndImmersion(
   double magnification, ImmersionType immersion)
{
   std::vector<const LensInfo*> result;
   for (int i = 0; i < LENS_DATABASE_SIZE; i++)
   {
      if (LENS_DATABASE[i].magnification == magnification &&
          LENS_DATABASE[i].immersion == immersion)
      {
         result.push_back(&LENS_DATABASE[i]);
      }
   }
   return result;
}

// Helper function: Find lens by model name
inline const LensInfo* GetLensByModel(const char* model)
{
   for (int i = 0; i < LENS_DATABASE_SIZE; i++)
   {
      if (strcmp(LENS_DATABASE[i].model, model) == 0)
      {
         return &LENS_DATABASE[i];
      }
   }
   return nullptr;
}

// Helper function: Find lens by model code
inline const LensInfo* GetLensByModelCode(unsigned int modelCode)
{
   for (int i = 0; i < LENS_DATABASE_SIZE; i++)
   {
      if (LENS_DATABASE[i].modelCode == modelCode)
      {
         return &LENS_DATABASE[i];
      }
   }
   return nullptr;
}

} // namespace EvidentLens
