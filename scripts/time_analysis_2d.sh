#!/bin/bash

echo "---2D---"
echo "With linear interpolation"
echo ""

echo "SSD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric ssd -repetitions 50 -iterations 1000 -interpolation linear -dim 2
echo "NCC"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric ncc -repetitions 50 -iterations 1000 -interpolation linear -dim 2
echo "MI"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric mi -repetitions 50 -iterations 1000 -interpolation linear -dim 2
echo "AlphaSMD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric alpha_smd -repetitions 50 -iterations 1000 -interpolation linear -dim 2

echo "---2D---"
echo "With cubic interpolation"
echo ""

echo "SSD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric ssd -repetitions 50 -iterations 1000 -interpolation cubic -dim 2
echo "NCC"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric ncc -repetitions 50 -iterations 1000 -interpolation cubic -dim 2
echo "MI"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png -in2 /home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/04.png -metric mi -repetitions 50 -iterations 1000 -interpolation cubic -dim 2
