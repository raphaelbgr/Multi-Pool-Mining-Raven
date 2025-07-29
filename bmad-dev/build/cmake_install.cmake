# Install script for directory: C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/BMAD_MultiPool_KawPow")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/Debug/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/Release/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/MinSizeRel/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/RelWithDebInfo/bmad_kawpow_multi.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/bin/Debug/bmad_kawpow_multi.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/bin/Release/bmad_kawpow_multi.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/bin/MinSizeRel/bmad_kawpow_multi.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/bin/RelWithDebInfo/bmad_kawpow_multi.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/lib/Debug/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/lib/Release/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/lib/MinSizeRel/bmad_kawpow_multi.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/lib/RelWithDebInfo/bmad_kawpow_multi.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/bmad" TYPE FILE FILES
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_kawpow_multi.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_memory_manager.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_pool_manager.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_types.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_agent_manager.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_kawpow_algorithm.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_kawpow_optimized.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_gpu_memory_manager.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_performance_tester.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_xmrig_bridge.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_share_converter.h"
    "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/include/bmad_pool_connector.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/test/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/rbgnr/git/Fork-Raven-Cuda-Minder/bmad-dev/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
