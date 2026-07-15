# - Find Cityhash
# 
# Copyright (c) 2019 jinpengliu@163.com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set(CITYHASH_USE_STATIC_LIBS true)

# Set the library prefix and library suffix properly.
if(CITYHASH_USE_STATIC_LIBS)
	set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_STATIC_LIBRARY_PREFIX})
	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
	set(LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
	set(LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
	set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_SHARED_LIBRARY_PREFIX})
	set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
	set(LIBRARY_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
	set(LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

include(FindPackageHandleStandardArgs)

macro(DO_FIND_CITYHASH_SYSTEM)
	find_path(CITYHASH_INCLUDE_DIR city.h
		PATHS /usr/local/include /usr/include
		)
	find_library(CITYHASH_LIBRARY
		NAMES cityhash
		PATHS /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64
		)
	FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cityhash DEFAULT_MSG
		CITYHASH_INCLUDE_DIR CITYHASH_LIBRARY
		)
	set(CITYHASH_LIBRARIES ${CITYHASH_LIBRARY})
	set(CITYHASH_INCLUDE_DIRS ${CITYHASH_INCLUDE_DIR})
	get_filename_component(CITYHASH_LIB_DIR ${CITYHASH_LIBRARY} DIRECTORY)
	set(CITYHASH_LIB_DIRS ${CITYHASH_LIB_DIR})
	mark_as_advanced(CITYHASH_LIBRARIES CITYHASH_INCLUDE_DIRS CITYHASH_LIB_DIRS)
endmacro()

macro(DO_FIND_CITYHASH_ROOT)
	if(NOT CITYHASH_ROOT_DIR)
		message(STATUS "CITYHASH_ROOT_DIR is not defined, using binary directory.")
		set(CITYHASH_ROOT_DIR ${CURRENT_CMAKE_BINARY_DIR} CACHE PATH "")
	endif()

	find_path(CITYHASH_INCLUDE_DIR city.h ${CITYHASH_ROOT_DIR}/include)
	find_library(CITYHASH_LIBRARY cityhash HINTS ${CITYHASH_ROOT_DIR}/lib)
	FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cityhash DEFAULT_MSG
		CITYHASH_INCLUDE_DIR CITYHASH_LIBRARY
		)
	set(CITYHASH_LIBRARIES ${CITYHASH_LIBRARY})
	set(CITYHASH_INCLUDE_DIRS ${CITYHASH_INCLUDE_DIR})
	get_filename_component(CITYHASH_LIB_DIR ${CITYHASH_LIBRARY} DIRECTORY)
	set(CITYHASH_LIB_DIRS ${CITYHASH_LIB_DIR})
	mark_as_advanced(CITYHASH_LIBRARIES CITYHASH_INCLUDE_DIRS CITYHASH_LIB_DIRS)
endmacro()

if(NOT CITYHASH_FOUND)
	DO_FIND_CITYHASH_ROOT()
endif()

if(NOT CITYHASH_FOUND)
	DO_FIND_CITYHASH_SYSTEM()
endif()