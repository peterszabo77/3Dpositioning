# 3Dpositioning

A study about how to obtain location coordinates based on 2D camera snapshots in a 3D space.

## About

The task is to locate our, i.e. the camera, position based on the 2D image that we see. The 3D scene is a virtual room with four different images on its four walls (the floor and ceiling are painted with homogeneous colors). The image of the camera is therefore a 2D projection of the 3D scene from a particular camera location and rotation angle (pitch is fixed).

## Solution

- supervised learning using a convolutional neural network 
- the inputs are 2D projection images with the camera locations (x and y coordinates) as labels 

## Software / libraries
Python, pyTorch, pyOpenGL
