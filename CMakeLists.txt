cmake_minimum_required(VERSION 3.16)
project(PotterySegmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

add_executable(pottery_segmentation pottery_segmentation.cpp)

target_link_libraries(pottery_segmentation
    OpenGL::GL
    glfw
    GLEW::GLEW
    glm::glm
)

# macOS: OpenGLの非推奨警告を抑制
if(APPLE)
    target_compile_definitions(pottery_segmentation PRIVATE GL_SILENCE_DEPRECATION)
endif()
