# Image Mosaic and Inpainting

## Overview
This repository, `image_mosaic`, is dedicated to exploring and implementing advanced optimization models to tackle two specific imaging tasks: inpainting and mosaic generation. Our project aims to apply methods introduced in academic lectures to solve these complex image processing challenges.

### Objectives
- **Inpainting**: Address areas of images that are damaged or missing, restoring them by intelligently filling in the gaps based on the information available in the surrounding areas.
- **Mosaic Generation**: Create high-quality mosaics from images by segmenting the image into a grid of pieces that are then recomposed to highlight certain artistic aspects.

## Problems
This project investigates two main classes of imaging problems:

1. **Image Inpainting**: This task involves techniques for reconstructing lost or deteriorated parts of images and videos. The goal is to restore images in a way that is seamless and undetectable to an ordinary observer.

2. **Mosaic Generation**: In this task, the aim is to combine various pieces of images (potentially from multiple sources) into a single cohesive image that represents a new, artistic interpretation of the combined visuals.

## Features
- **Convex Optimization Problems**: Developed models to tackle total variation minimization and sparse reconstruction.
- **Application**: Applied these models to damaged images to reconstruct the missing information effectively.

- **Linear Convex Optimization**: Created linear models to generate image mosaics from grayscale sources.
- **Dynamic Sizing**: Enhanced the models to adjust mosaic sizes, enabling the generation of various image outputs based on the same input.

- **Model Adaptation**: Extended the linear convex models to work with colored images.
- **Colored Mosaics**: Generated colored mosaic images that correspond to the original images by effectively solving the optimization challenges posed.

## Run and Test
To get started with this project, clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/image_mosaic.git
cd image_mosaic
# Setup your MATLAB environment (if necessary)
# Run and test part1_1.m 
# Run and test part1_2A.m and part1_2B.m
# Run and test part2.m
# Check the generated images against the images in the results
```


