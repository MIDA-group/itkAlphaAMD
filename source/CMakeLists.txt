cmake_minimum_required(VERSION 3.5)
project(ACMetric)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(ITK_DIR "c:/dev/itk-sandbox/ITK-build/")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

SET(HEADERS
  transforms/itkAlphaSMDAffineTransform.h
  metric/itkAlphaSMDMetric2.h
  metric/itkAlphaSMDMetricInternal2.h
  registration/registrationRoutines.h
)

add_executable(ACRandomTransforms acRandomTransforms.cpp ${HEADERS})

add_executable(ACRegister acRegister.cpp ${HEADERS})

add_executable(ACTransform acTransform.cpp)

add_executable(ACTransformLandmarks acTransformLandmarks.cpp)

add_executable(ACLabelOverlap acLabelOverlap.cpp)

add_executable(FormatConv formatConv.cpp)

add_executable(ACTimeAnalysis acTimeAnalysis.cpp)

target_link_libraries(ACRandomTransforms ${ITK_LIBRARIES})

target_link_libraries(ACRegister ${ITK_LIBRARIES})

target_link_libraries(ACTransform ${ITK_LIBRARIES})

target_link_libraries(ACTransformLandmarks ${ITK_LIBRARIES})

target_link_libraries(ACLabelOverlap ${ITK_LIBRARIES})

target_link_libraries(FormatConv ${ITK_LIBRARIES})

target_link_libraries(ACTimeAnalysis ${ITK_LIBRARIES})
