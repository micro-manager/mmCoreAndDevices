set(module_name EnderscopeStage)

set(module_srcs
  EnderscopeStage.cpp
)

set(module_hdrs
  EnderscopeStage.h
)

# In mmCoreAndDevices, this target is usually created by helper macros in parent CMake.
# If your parent build uses a different pattern, adapt this block accordingly.
add_library(${module_name} MODULE ${module_srcs} ${module_hdrs})

target_include_directories(${module_name}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/MMDevice
)
