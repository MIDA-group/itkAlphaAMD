#!/bin/bash

echo "---2D---"
echo "With linear interpolation"
echo ""

echo "AlphaSMD"
/home/johof680/work/itkAlphaAMD-build/ACTimeAnalysis -in1 ../images/Bronchiolar_area_cilia_cross-sections_1.jpg -in2 ../images/Bronchiolar_area_cilia_cross-sections_1_flipped.jpg -metric alpha_smd -repetitions 1000 -iterations 1 -interpolation linear -dim 2

echo "---2D---"
echo "With cubic interpolation"
echo ""

echo "SSD"
/home/johof680/work/itkAlphaAMD-build/ACTimeAnalysis -in1 ../images/Bronchiolar_area_cilia_cross-sections_1.jpg -in2 ../images/Bronchiolar_area_cilia_cross-sections_1_flipped.jpg -metric ssd -repetitions 1000 -iterations 1 -interpolation cubic -dim 2
echo "NCC"
/home/johof680/work/itkAlphaAMD-build/ACTimeAnalysis -in1 ../images/Bronchiolar_area_cilia_cross-sections_1.jpg -in2 ../images/Bronchiolar_area_cilia_cross-sections_1_flipped.jpg -metric ncc -repetitions 1000 -iterations 1 -interpolation cubic -dim 2
echo "MI"
/home/johof680/work/itkAlphaAMD-build/ACTimeAnalysis -in1 ../images/Bronchiolar_area_cilia_cross-sections_1.jpg -in2 ../images/Bronchiolar_area_cilia_cross-sections_1_flipped.jpg -metric mi -repetitions 1000 -iterations 1 -interpolation cubic -dim 2
