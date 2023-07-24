# Define the source files and executables
set(TEST_SOURCES
    FloatPropertyTruncation-Tests.cpp
    MMTime-Tests.cpp
)

add_executable(FloatPropertyTruncation-Tests FloatPropertyTruncation-Tests.cpp)
add_executable(MMTime-Tests MMTime-Tests.cpp)

# Include directories and linker flags
target_include_directories(FloatPropertyTruncation-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)
target_include_directories(MMTime-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)

# Link libraries
target_link_libraries(FloatPropertyTruncation-Tests gmock libMMDevice)
target_link_libraries(MMTime-Tests gmock libMMDevice)

# Define tests
add_test(NAME FloatPropertyTruncation-Tests COMMAND FloatPropertyTruncation-Tests)
add_test(NAME MMTime-Tests COMMAND MMTime-Tests)
