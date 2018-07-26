#!/bin/bash

echo "---3D---"
echo "Linear Interpolation"
echo "SSD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric ssd -repetitions 50 -iterations 1 -interpolation linear -dim 3
echo "NCC"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric ncc -repetitions 50 -iterations 1 -interpolation linear -dim 3
echo "MI"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric mi -repetitions 50 -iterations 1 -interpolation linear -dim 3
echo "AlphaSMD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric alpha_smd -repetitions 50 -iterations 1 -interpolation linear -dim 3

echo ""
echo "Cubic Interpolation"
echo "SSD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric ssd -repetitions 50 -iterations 1 -interpolation cubic -dim 3
echo "NCC"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric ncc -repetitions 50 -iterations 1 -interpolation cubic -dim 3
echo "MI"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric mi -repetitions 50 -iterations 1 -interpolation cubic -dim 3
echo "AlphaSMD"
/home/johof680/work/itkAlphaCut-4j/build-release/ACTimeAnalysis -in1 /home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz -in2 /home/johof680/work/LPBA40_for_ANTs/mri/S02.nii.gz -metric alpha_smd -repetitions 50 -iterations 1 -interpolation cubic -dim 3