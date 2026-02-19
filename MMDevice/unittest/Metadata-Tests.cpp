#include <catch2/catch_all.hpp>

#include "ImageMetadata.h"
#include "CameraImageMetadata.h"

#include <string>

// --- CameraImageMetadata serialization ---

TEST_CASE("CameraImageMetadata serialize empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   CHECK(std::string(m.Serialize()) == "0\n");
}

// --- Legacy Metadata serialization via PutImageTag() ---

TEST_CASE("Legacy Metadata serialize via PutImageTag empty", "[Metadata]") {
   Metadata md;
   CHECK(md.Serialize() == "0\n");
}

TEST_CASE("Legacy Metadata serialize via PutImageTag single string tag",
          "[Metadata]") {
   Metadata md;
   md.PutImageTag("Exposure", "10.0");
   CHECK(md.Serialize() == "1\ns\nExposure\n_\n1\n10.0\n");
}

TEST_CASE("Legacy Metadata serialize via PutImageTag three tags",
          "[Metadata]") {
   Metadata md;
   md.PutImageTag("Exposure", "10.0");
   md.PutImageTag("Binning", 2);
   md.PutImageTag("X", 1.5);
   CHECK(md.Serialize() ==
       "3\n"
       "s\nBinning\n_\n1\n2\n"
       "s\nExposure\n_\n1\n10.0\n"
       "s\nX\n_\n1\n1.5\n");
}

// --- CameraImageMetadata to legacy Metadata restore ---

TEST_CASE("CameraImageMetadata to legacy Metadata restore empty",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   Metadata md;
   REQUIRE(md.Restore(m.Serialize()));
   CHECK(md.GetKeys().empty());
}

TEST_CASE("CameraImageMetadata to legacy Metadata restore single string tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   Metadata md;
   REQUIRE(md.Restore(m.Serialize()));
   CHECK(md.GetKeys().size() == 1);
   CHECK(md.HasTag("Exposure"));
   auto tag = md.GetSingleTag("Exposure");
   CHECK(tag.GetName() == "Exposure");
   CHECK(tag.GetDevice() == "_");
   CHECK(tag.GetValue() == "10.0");
}

TEST_CASE("CameraImageMetadata to legacy Metadata restore single int tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Binning", 2);
   Metadata md;
   REQUIRE(md.Restore(m.Serialize()));
   CHECK(md.GetKeys().size() == 1);
   CHECK(md.HasTag("Binning"));
   auto tag = md.GetSingleTag("Binning");
   CHECK(tag.GetName() == "Binning");
   CHECK(tag.GetDevice() == "_");
   CHECK(tag.GetValue() == "2");
}

TEST_CASE("CameraImageMetadata to legacy Metadata restore single double tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("X", 1.5);
   Metadata md;
   REQUIRE(md.Restore(m.Serialize()));
   CHECK(md.GetKeys().size() == 1);
   CHECK(md.HasTag("X"));
   auto tag = md.GetSingleTag("X");
   CHECK(tag.GetName() == "X");
   CHECK(tag.GetDevice() == "_");
   CHECK(tag.GetValue() == "1.5");
}

TEST_CASE("CameraImageMetadata to legacy Metadata restore three tags",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.AddTag("X", 1.5);
   Metadata md;
   REQUIRE(md.Restore(m.Serialize()));
   CHECK(md.GetKeys().size() == 3);

   CHECK(md.HasTag("Binning"));
   auto tag1 = md.GetSingleTag("Binning");
   CHECK(tag1.GetName() == "Binning");
   CHECK(tag1.GetDevice() == "_");
   CHECK(tag1.GetValue() == "2");

   CHECK(md.HasTag("Exposure"));
   auto tag2 = md.GetSingleTag("Exposure");
   CHECK(tag2.GetName() == "Exposure");
   CHECK(tag2.GetDevice() == "_");
   CHECK(tag2.GetValue() == "10.0");

   CHECK(md.HasTag("X"));
   auto tag3 = md.GetSingleTag("X");
   CHECK(tag3.GetName() == "X");
   CHECK(tag3.GetDevice() == "_");
   CHECK(tag3.GetValue() == "1.5");
}

// --- Legacy Metadata PutImageTag() and CameraImageMetadata migration ---

// These tests show that CameraImageMetadata produces equivalent results to
// legacy Metadata::PutImageTag(), by restoring both into Metadata and
// comparing tag values.

TEST_CASE("CameraImageMetadata and legacy PutImageTag produce same tags, empty",
          "[Metadata]")
{
   MM::CameraImageMetadata m;
   Metadata legacyMd;

   Metadata fromNew;
   REQUIRE(fromNew.Restore(m.Serialize()));
   Metadata fromLegacy;
   REQUIRE(fromLegacy.Restore(legacyMd.Serialize().c_str()));

   CHECK(fromNew.GetKeys() == fromLegacy.GetKeys());
}

TEST_CASE("CameraImageMetadata and legacy PutImageTag produce same tags, string",
          "[Metadata]")
{
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");

   Metadata legacyMd;
   legacyMd.PutImageTag("Exposure", "10.0");

   Metadata fromNew;
   REQUIRE(fromNew.Restore(m.Serialize()));
   Metadata fromLegacy;
   REQUIRE(fromLegacy.Restore(legacyMd.Serialize().c_str()));

   CHECK(fromNew.GetKeys() == fromLegacy.GetKeys());
   CHECK(fromNew.GetSingleTag("Exposure").GetValue() ==
         fromLegacy.GetSingleTag("Exposure").GetValue());
}

TEST_CASE("CameraImageMetadata and legacy PutImageTag produce same tags, int",
          "[Metadata]")
{
   MM::CameraImageMetadata m;
   m.AddTag("Binning", 2);

   Metadata legacyMd;
   legacyMd.PutImageTag("Binning", 2);

   Metadata fromNew;
   REQUIRE(fromNew.Restore(m.Serialize()));
   Metadata fromLegacy;
   REQUIRE(fromLegacy.Restore(legacyMd.Serialize().c_str()));

   CHECK(fromNew.GetKeys() == fromLegacy.GetKeys());
   CHECK(fromNew.GetSingleTag("Binning").GetValue() ==
         fromLegacy.GetSingleTag("Binning").GetValue());
}

TEST_CASE("CameraImageMetadata and legacy PutImageTag produce same tags, double",
          "[Metadata]")
{
   MM::CameraImageMetadata m;
   m.AddTag("X", 1.5);

   Metadata legacyMd;
   legacyMd.PutImageTag("X", 1.5);

   Metadata fromNew;
   REQUIRE(fromNew.Restore(m.Serialize()));
   Metadata fromLegacy;
   REQUIRE(fromLegacy.Restore(legacyMd.Serialize().c_str()));

   CHECK(fromNew.GetKeys() == fromLegacy.GetKeys());
   CHECK(fromNew.GetSingleTag("X").GetValue() ==
         fromLegacy.GetSingleTag("X").GetValue());
}

TEST_CASE("CameraImageMetadata and legacy PutImageTag produce same tags, three tags",
          "[Metadata]")
{
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.AddTag("X", 1.5);

   Metadata legacyMd;
   legacyMd.PutImageTag("Exposure", "10.0");
   legacyMd.PutImageTag("Binning", 2);
   legacyMd.PutImageTag("X", 1.5);

   Metadata fromNew;
   REQUIRE(fromNew.Restore(m.Serialize()));
   Metadata fromLegacy;
   REQUIRE(fromLegacy.Restore(legacyMd.Serialize().c_str()));

   CHECK(fromNew.GetKeys() == fromLegacy.GetKeys());
   for (const auto& key : fromNew.GetKeys()) {
      CHECK(fromNew.GetSingleTag(key.c_str()).GetValue() ==
            fromLegacy.GetSingleTag(key.c_str()).GetValue());
   }
}
