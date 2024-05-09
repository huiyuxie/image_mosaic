# Image Mosaic and Inpainting Project

## Overview
This repository, `image_mosaic`, is dedicated to exploring and implementing advanced optimization models to tackle two specific imaging tasks: inpainting and mosaic generation. Our project aims to apply minimization methodologies introduced in academic lectures to solve these complex image processing challenges.

### Objectives
- **Inpainting**: Address areas of images that are damaged or missing, restoring them by intelligently filling in the gaps based on the information available in the surrounding areas.
- **Mosaic Generation**: Create high-quality mosaics from images by segmenting the image into a grid of pieces that are then recomposed to highlight certain artistic aspects.

## Problem Classes
This project investigates two main classes of imaging problems:

1. **Inpainting**: This task involves techniques for reconstructing lost or deteriorated parts of images and videos. The goal is to restore images in a way that is seamless and undetectable to an ordinary observer.

2. **Mosaic Generation**: In this task, the aim is to combine various pieces of images (potentially from multiple sources) into a single cohesive image that represents a new, artistic interpretation of the combined visuals.

## Methodology
The methodologies employed in this project are derived from optimization models discussed in lectures. These include:

- Gradient descent methods
- Convex and non-convex optimization approaches
- Heuristic algorithms for efficient computation

## Installation
To get started with this project, clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/image_mosaic.git
cd image_mosaic
# Setup your environment (if necessary)
