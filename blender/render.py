import bpy
import math
import os
from tqdm import tqdm

PATH = os.path.dirname(bpy.data.filepath) + "/"
FORMAT = "PNG"
MASK_SET = False

info = {
    "sun_intensity": 0.8,
    "sun_elevation": 90.0,
    "sun_rotation":  180.0,
    "strength":      0.1
}

if MASK_SET:
    assert bpy.context.scene.render.engine == "BLENDER_EEVEE"
    output_path = PATH + "masks/"

else:
    assert bpy.context.scene.render.engine == "CYCLES"
    output_path = PATH + "renders/"

    sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
    sky.sun_intensity = info["sun_intensity"]
    sky.sun_elevation = math.radians(info["sun_elevation"])
    sky.sun_rotation = math.radians(info["sun_rotation"])

    bpy.data.worlds["sky"].node_tree.nodes["Background"].inputs[1].default_value = info["strength"]

scene = bpy.context.scene

start_frame = scene.frame_start
end_frame = scene.frame_end

for frame in tqdm(range(start_frame, end_frame + 1)):
    scene.frame_set(frame)
    bpy.ops.render.render(write_still=True)

    if MASK_SET:
        filename = f"keyboard_{frame:03d}.{FORMAT.lower()}"
    else:
        filename = f"keyboard_{info}_{frame:03d}.{FORMAT.lower()}"

    bpy.data.images["Render Result"].save_render(output_path + filename)
