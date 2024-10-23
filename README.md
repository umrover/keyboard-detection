# Keyboard Key Detection

This project aims to create an ML pipeline to generate synthetic training images using Blender, create realistic data
augmentation, and fine-tune various image segmentation models.

The goal of the models is to detect:

1. the location of keys (key regions) on a keyboard
2. the letter/symbol they correspond to

## Dataset Generation

A Blender file with a 3D modelled keyboard and camera is provided in [keyboard.blend](blender/keyboard.blend).

First, [setup_viewpoints.py](blender/setup_viewpoints.py) places the camera along the vertices of the Viewpoints object
and orients it to face the keyboard. The camera's position and rotation is randomly perturbed at each keyframe. To
define additional viewpoints, edit the Viewpoints object.

![blender.png](assets/docs/blender.png)

Next, [segmentation_mask.py](blender/segmentation_mask.py) provides a script to automatically assign flat colors to each
key's face. Set `SET_MASK = True` to create the segmentation masks, or `SET_MASK = False` to revert back.

To find key faces, the program iterates through each face in the 'Keys' collection and checks if its dot product with
the orientation vector (the normal of the 'Orientation' object) is small.

![masks_generation.png](assets/docs/masks_generation.png)

Then, using [render.py](blender/render.py), sky lightning conditions are configured and several keyboard angles are
rendered using Cycles. Masks are rendered using Eevee.

![keyboard_render.png](assets/docs/keyboard_034.png)

Finally, the image is imported into Python and the process in [data-augmentation.ipynb](data-augmentation.ipynb) is
followed. A random background is chosen from a dataset of 4000 landscapes.

**_Note: dataset v1 did not include shadows._** 

To enhance realism and robustness to lighting conditions, we augment the images with various shadow and sunlight effects. First, a straight solid line is drawn over a transparent mask. Its thickness is sampled uniformly between 100-400 pixels. Then, a random affine transform is applied.

This produced a hard alpha mask. A box blur is applied with a kernel size `k ~ max(5, N(μ=25, σ=10))`. This produced a soft alpha mask.

We choose a sunlight/shadow effect randomly. Exposure is `e ~ N(μ=2.5, σ=1)` for sunlight and `e ~ N(μ=0.5, σ=0.5)` for shadow. The exposure is clamped to be greater than 0.4. Then, the image's exposure is edited and merged with the original using the alpha mask.

| Alpha Mask                      | Sunlight                           | Shadow                           |
|---------------------------------|------------------------------------|----------------------------------|
| ![](assets/docs/alpha_mask.png) | ![](assets/docs/sunlight_mask.png) | ![](assets/docs/shadow_mask.png) |

For further augmentation, the number of lighting effects is sampled from a geometric distribution and the above process is repeated. This often produces complex and beautiful lighting effects.

![shadow.png](assets/docs/shadows.png)

Then, a motion blur along a random direction (0° to 360°) + random kernel size (4-16) is applied along with a vignette
of random strength.

Finally, contrast, exposure, sharpness, saturation are sampled from a Gaussian distribution (μ=1, σ=0.5), clamped, and
applied to the image.

![keyboard.png](assets/docs/keyboard.png)

To speed up dataset creation, a multiprocessing pipeline is defined in [create_dataset.py](datasets/create_dataset.py)

![keyboard_dataset.png](assets/docs/keyboard_dataset.png)
