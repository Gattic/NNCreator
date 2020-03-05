#Import libs
find_package(shmea REQUIRED)
find_package(glades REQUIRED)
find_package(gfxplusplus REQUIRED)

#Subdirectories
include_directories(${CMAKE_SOURCE_DIR})
include_directories("services")
add_subdirectory("core")

set(MAIN_src_files
	crt0.cpp
	crt0.h
	main.cpp
	main.h
	nncreator.cpp
	nncreator.h
)
add_executable(${PROJECT_NAME} ${MAIN_src_files})

#Link libraries
target_link_libraries(${PROJECT_NAME}
	core gfxplusplus glades shmea)#custom libs

target_include_directories(${PROJECT_NAME} PRIVATE "include")
