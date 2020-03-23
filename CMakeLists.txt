cmake_minimum_required(VERSION 3.5)
project(ACMetric)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(ITK_DIR "c:/dev/itk-sandbox/ITK-build/")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

SET(HEADERS
  source/transforms/itkAlphaSMDAffineTransform.h
  source/metric/itkAlphaSMDMetric2.h
  source/metric/itkAlphaSMDMetricInternal2.h
  source/registration/registrationRoutines.h
)

add_executable(ACRandomTransforms source/acRandomTransforms.cpp ${HEADERS})

add_executable(ACRegister source/acRegister.cpp ${HEADERS})

add_executable(ACTransform source/acTransform.cpp)

add_executable(ACTransformLandmarks source/acTransformLandmarks.cpp)

add_executable(ACLabelOverlap source/acLabelOverlap.cpp)

add_executable(FormatConv source/formatConv.cpp)

add_executable(ACTimeAnalysis source/acTimeAnalysis.cpp)

add_executable(ACTestBSpline source/acTestBSpline.cpp)

add_executable(RegisterBSpline3D source/registerBSpline3D_2.cpp)

add_executable(SynthEvalDeformableND source/synthEvalDeformableND.cpp)

add_executable(PWEvalDeformableND source/pwEvalDeformableND.cpp)

add_executable(ABSReg source/absReg.cpp)

add_executable(TestMCDS source/testMCDS.cpp)

add_executable(TestSuite source/tests/testSuite.cpp)

add_executable(RegisterDeformable source/programs/registerDeformable.cpp)

add_executable(EvaluateRegistration source/programs/evaluateRegistration.cpp)

target_link_libraries(ACRandomTransforms ${ITK_LIBRARIES})

target_link_libraries(ACRegister ${ITK_LIBRARIES})

target_link_libraries(ACTransform ${ITK_LIBRARIES})

target_link_libraries(ACTransformLandmarks ${ITK_LIBRARIES})

target_link_libraries(ACLabelOverlap ${ITK_LIBRARIES})

target_link_libraries(FormatConv ${ITK_LIBRARIES})

target_link_libraries(ACTimeAnalysis ${ITK_LIBRARIES})

target_link_libraries(ACTestBSpline ${ITK_LIBRARIES})

target_link_libraries(RegisterBSpline3D ${ITK_LIBRARIES})

target_link_libraries(SynthEvalDeformableND ${ITK_LIBRARIES})

target_link_libraries(PWEvalDeformableND ${ITK_LIBRARIES})

target_link_libraries(ABSReg ${ITK_LIBRARIES})

target_link_libraries(TestMCDS ${ITK_LIBRARIES})

target_link_libraries(TestSuite ${ITK_LIBRARIES})

target_link_libraries(RegisterDeformable ${ITK_LIBRARIES})

target_link_libraries(EvaluateRegistration ${ITK_LIBRARIES})
