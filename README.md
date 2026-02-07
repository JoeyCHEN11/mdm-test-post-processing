# Naive Self-Intersect Correction of MDM Results

This is a demo project for post-processing of coap on Human Motion Diffusion results.


## Requirements

- .npy results with [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
    - MDM always uses beta=0 and the gender-neutral model
- [SMPLX](https://smpl-x.is.tue.mpg.de/) body models
- Currently, we use [COAP](https://github.com/markomih/COAP) for self-intersection detection

## Results

Because we just naively corrected each frame of the motion, the result is not such continuous. However, the self-intersection can be resolved efficiently.

### Before

- Motion: Squat

  ![Squat](./image/README/squat_before.gif)

- Motion: Cross arms

  ![Hug](./image/README/cross_before.gif)

### After

- Motion: Squat

  ![Squat](./image/README/squat_after.gif)

- Motion: Cross arms

  ![Hug](./image/README/cross_after.gif)
