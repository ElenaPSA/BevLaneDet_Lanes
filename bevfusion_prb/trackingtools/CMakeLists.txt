cmake_minimum_required(VERSION 2.8.12)
project(TrackingUtilsPython)
set(CMAKE_CXX_STANDARD 11)
add_subdirectory(pybind11)

MESSAGE( " *** PYTHON_EXECUTABLE : " ${PYTHON_EXECUTABLE} )
MESSAGE( " *** PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS} )
MESSAGE( " *** PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES} )

find_package(OpenCV REQUIRED)
MESSAGE( " *** OpenCV_INCLUDE_DIRS : " ${OpenCV_INCLUDE_DIRS} )
MESSAGE( " *** OpenCV_LIBS : " ${OpenCV_LIBS} )
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")
# Get numpy include dir
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
			  RESULT_VARIABLE _numpy_process
			  OUTPUT_VARIABLE Numpy_INCLUDE_DIRS
			  OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE( " *** Numpy_INCLUDE_DIRS : " ${Numpy_INCLUDE_DIRS} )

include_directories(${Numpy_INCLUDE_DIRS})

add_subdirectory(TrackingLib)

include_directories (${TrackingUtilsLibrary_INCLUDE_DIRS})
# Create Python module
pybind11_add_module(TrackingUtilsPy pythonpackage/ndarray_converter.cpp
									pythonpackage/ndarray_converter.h
									pythonpackage/python_TrackSmoothing.cpp
									pythonpackage/python_module.cpp)
									
target_link_libraries(TrackingUtilsPy PRIVATE ${OpenCV_LIBS} TrackingUtilsLibrary)

install(TARGETS  TrackingUtilsPy DESTINATION "bin")
install(FILES	"${CMAKE_CURRENT_SOURCE_DIR}/python/example.py"
				DESTINATION "bin")
