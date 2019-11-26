
#include "testPointSamplers.cpp"
#include "testValueSamplers.cpp"

int main(int argc, char** argv) {
    PointSamplerTests::RunPointSamplersTests();
    ValueSamplerTests::RunValueSamplersTests();

    return 0;
}