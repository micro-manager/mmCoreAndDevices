#include <catch2/catch_all.hpp>

#include "CameraImageMetadata.h"
#include "ImageMetadata.h"
#include "SerializedMetadata.h"

#include <string>

using mmcore::internal::SerializedMetadata;

namespace {
bool RestoreFromView(Metadata& md, std::string_view view) {
   return md.Restore(std::string(view).c_str());
}
} // namespace

TEST_CASE("Empty SerializedMetadata round-trips through Metadata::Restore") {
   SerializedMetadata sm;
   Metadata md;
   REQUIRE(RestoreFromView(md, sm.View()));
   CHECK(md.GetKeys().empty());
}

TEST_CASE("AddTag values are visible after Metadata::Restore") {
   SerializedMetadata sm;
   sm.AddTag("Greeting", "hello");
   sm.AddTag("IntVal", 42);
   sm.AddTag("FloatVal", 3.5);

   Metadata md;
   REQUIRE(RestoreFromView(md, sm.View()));
   CHECK(md.GetSingleTag("Greeting").GetValue() == "hello");
   CHECK(md.GetSingleTag("IntVal").GetValue() == "42");
   CHECK(md.GetSingleTag("FloatVal").GetValue() == "3.5");
   CHECK(md.GetKeys().size() == 3);
}

TEST_CASE("Construction from CameraImageMetadata blob preserves tags") {
   MM::CameraImageMetadata cim;
   cim.AddTag("A", "1");
   cim.AddTag("B", "two");

   SerializedMetadata sm(cim.Serialize());
   sm.AddTag("C", "three");

   Metadata md;
   REQUIRE(RestoreFromView(md, sm.View()));
   CHECK(md.GetSingleTag("A").GetValue() == "1");
   CHECK(md.GetSingleTag("B").GetValue() == "two");
   CHECK(md.GetSingleTag("C").GetValue() == "three");
   CHECK(md.GetKeys().size() == 3);
}

TEST_CASE("Construction from null or empty produces empty buffer") {
   SerializedMetadata smNull(nullptr);
   SerializedMetadata smEmpty("");
   Metadata md;
   REQUIRE(RestoreFromView(md, smNull.View()));
   CHECK(md.GetKeys().empty());
   REQUIRE(RestoreFromView(md, smEmpty.View()));
   CHECK(md.GetKeys().empty());
}

TEST_CASE("HasTag and GetTag find added tags") {
   SerializedMetadata sm;
   sm.AddTag("foo", "1");
   sm.AddTag("bar", "two");

   CHECK(sm.HasTag("foo"));
   CHECK(sm.HasTag("bar"));
   CHECK_FALSE(sm.HasTag("baz"));

   REQUIRE(sm.GetTag("foo").has_value());
   CHECK(*sm.GetTag("foo") == "1");
   REQUIRE(sm.GetTag("bar").has_value());
   CHECK(*sm.GetTag("bar") == "two");
   CHECK_FALSE(sm.GetTag("baz").has_value());
}

TEST_CASE("HasTag does not match substrings of keys or values") {
   SerializedMetadata sm;
   sm.AddTag("foobar", "value");
   sm.AddTag("other", "containsfoo");

   CHECK(sm.HasTag("foobar"));
   CHECK_FALSE(sm.HasTag("foo"));
   CHECK_FALSE(sm.HasTag("bar"));
   CHECK_FALSE(sm.HasTag("contains"));
}

TEST_CASE("GetTag returns the last occurrence for duplicate keys") {
   SerializedMetadata sm;
   sm.AddTag("k", "first");
   sm.AddTag("k", "second");
   sm.AddTag("k", "third");

   REQUIRE(sm.GetTag("k").has_value());
   CHECK(*sm.GetTag("k") == "third");
}

TEST_CASE("AppendSerialized merges tags and updates the count") {
   MM::CameraImageMetadata cim;
   cim.AddTag("X", "1");
   cim.AddTag("Y", "2");

   SerializedMetadata sm;
   sm.AddTag("A", "a");
   sm.AppendSerialized(cim.Serialize());

   Metadata md;
   REQUIRE(RestoreFromView(md, sm.View()));
   CHECK(md.GetSingleTag("A").GetValue() == "a");
   CHECK(md.GetSingleTag("X").GetValue() == "1");
   CHECK(md.GetSingleTag("Y").GetValue() == "2");
   CHECK(md.GetKeys().size() == 3);
}

TEST_CASE("AppendSerialized handles a null pointer") {
   SerializedMetadata sm;
   sm.AddTag("A", "a");
   sm.AppendSerialized(nullptr);

   Metadata md;
   REQUIRE(RestoreFromView(md, sm.View()));
   CHECK(md.GetSingleTag("A").GetValue() == "a");
   CHECK(md.GetKeys().size() == 1);
}
