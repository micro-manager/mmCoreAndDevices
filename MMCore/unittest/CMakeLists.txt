# Define the source files and executables
set(TEST_SOURCES
    APIError-Tests.cpp
    CoreSanity-Tests.cpp
    LoggingSplitEntryIntoLines-Tests.cpp
    Logger-Tests.cpp
)

add_executable(APIError-Tests APIError-Tests.cpp)
add_executable(CoreSanity-Tests CoreSanity-Tests.cpp)
add_executable(LoggingSplitEntryIntoLines-Tests LoggingSplitEntryIntoLines-Tests.cpp)
add_executable(Logger-Tests Logger-Tests.cpp)

# Include directories and linker flags
target_include_directories(APIError-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)
target_include_directories(CoreSanity-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)
target_include_directories(LoggingSplitEntryIntoLines-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)
target_include_directories(Logger-Tests PRIVATE ${GMOCK_INCLUDE_DIRS} ..)

# Link libraries
target_link_libraries(APIError-Tests gmock libMMCore)
target_link_libraries(CoreSanity-Tests gmock libMMCore)
target_link_libraries
