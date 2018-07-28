
itkAlphaAMD is a registration framework (and distance measure) building on Insight Segmentation and Registration Toolkit.

Author: Johan Öfverstedt

Requirements/Dependencies:

ITK 4.x (tested with 4.9).
CMake

Tested on Windows 10, MacOSX and Ubuntu 16.04 LTS.

To build the framework (on Linux), go to the directory directly above the repository:

$ mkdir itkAlphaAMD-build/
$ cd itkAlphaAMD-build/
$ cmake -D CMAKE_BUILD_TYPE=Release ../itkAlphaAMD/source/
$ make

Now there will be a number of command line applications in the folder:

ACRegister - Tegisters pairs of images to a common space.
ACTransform - Warps images according to a given transformation, e.g. found by ACRegister.
ACTransformLandmarks - Transforms sets of landmarks given as a csv-file of coordinates according to a given transformation.
ACRandomTransfoms - Generates a set of image pairs subject to a specified class of transformations and noise level.
ACLabelOverlap - Computes the label overlap of label images.
FormatConv - Converts images between various formats.



