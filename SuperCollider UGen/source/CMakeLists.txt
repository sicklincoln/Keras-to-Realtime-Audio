#minimal Cmake file for SuperCollider plugin
#SC 3.5/3.4 compatible:
####cmake -DSC_PATH=/data/gitprojects/SuperCollider-Source3.5 -DCMAKE_OSX_ARCHITECTURES='i386;x86_64' ..
#cmake -DSC_PATH=/data/gitprojects/SuperCollider -DCMAKE_OSX_ARCHITECTURES='x86_64' ..

cmake_minimum_required (VERSION 2.6)

#https://crascit.com/2015/03/28/enabling-cxx11-in-cmake/
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (PV_Kerasify)

include_directories(${SC_PATH}/include/plugin_interface)
include_directories(${SC_PATH}/common)
include_directories(${SC_PATH}/include/common)
include_directories(${SC_PATH}/server/plugins) # for FFT_UGens.h
include_directories(${SC_PATH}/external_libraries/libsndfile/)
include_directories(${SC_PATH}/external_libraries/boost/)

set(CMAKE_SHARED_MODULE_PREFIX "")
if(APPLE OR WIN32)
set(CMAKE_SHARED_MODULE_SUFFIX ".scx")
endif()

add_library(PV_Kerasify MODULE keras_model.cc PV_Kerasify.cpp)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/PV_Kerasify.scx DESTINATION "/data/sc3code/plugins/PV_Kerasify/")
