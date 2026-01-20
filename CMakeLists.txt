if(NOT (${CMAKE_BINARY_DIR} STREQUAL "${CMAKE_SOURCE_DIR}/build"))
	message(FATAL_ERROR "Must be in \"build\" directory")
endif()

set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/include)

#Init
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#Compiler Flags
#set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE Debug) # -g

if(WIN32)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
else()
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # its set below
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#Project
project(NNCreator)
set(G_VERSION_MAJOR 0)
set(G_VERSION_MINOR 57)

# Backend build options
option(ENABLE_SDL2 "Build with SDL2 backend" OFF)
option(ENABLE_OPENGL "Build with OpenGL/GLFW backend" ON)

# Propagate compile definitions for conditional compilation
if(ENABLE_SDL2)
  add_definitions(-DGFX_HAVE_SDL2)
endif()
if(ENABLE_OPENGL)
  add_definitions(-DGFX_HAVE_OPENGL)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

#Import libs
find_package(shmea REQUIRED)
find_package(glades REQUIRED)
find_package(gfxplusplus REQUIRED)

#Subdirectories
include_directories(${CMAKE_SOURCE_DIR})
include_directories("include")
add_subdirectory(src)

#make run
add_custom_target(run
	COMMAND ${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make local
add_custom_target(local
	COMMAND ${PROJECT_NAME} local
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
#make docs
add_custom_target(docs
	COMMAND google-chrome index.html
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs/html/
)

#make fullscreen
add_custom_target(fullscreen
	COMMAND ${PROJECT_NAME} fullscreen
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make compat
add_custom_target(compat
	COMMAND ${PROJECT_NAME} compat
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make nogui
add_custom_target(nogui
	COMMAND ${PROJECT_NAME} nogui
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make debug
add_custom_target(debug
	COMMAND gdb ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make ddd
add_custom_target(ddd
	COMMAND ddd ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make mem
	add_custom_target(mem
	COMMAND valgrind --leak-check=full --track-origins=yes ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make profile
	add_custom_target(profile
	COMMAND valgrind --tool=callgrind ./build/${PROJECT_NAME}
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#make profilelocal
	add_custom_target(profilelocal
	COMMAND valgrind --tool=callgrind ./build/${PROJECT_NAME} local
	DEPENDS ${PROJECT_NAME}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
