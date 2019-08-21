
#include <iostream>
#include "synthEvalDeformable.cpp"

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "No arguments..." << std::endl;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        SynthEvalDeformable<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        SynthEvalDeformable<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
