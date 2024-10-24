import bpy
from mathutils import Vector

math_utils = bpy.data.texts["math_utils"].as_module()
blender_utils = bpy.data.texts["blender_utils"].as_module()

# PARAMETERS CONFIGURATION
# --------------------------------------------


SET_MASK = True

keyboard = 2

TOTAL_KEYBOARDS = 4
K = 2

# SCRIPT BEGINS HERE
# --------------------------------------------


for i in range(TOTAL_KEYBOARDS):
    objs = bpy.data.collections[f"Keyboard {i + 1}"]
    blender_utils.hide_collection(objs, True)

keyboard = bpy.data.collections[f"Keyboard {keyboard}"]
orientation = blender_utils.get_child_by_name(keyboard, "Orientation")
keys = blender_utils.get_children_by_name(keyboard, "Key")
keys += blender_utils.get_children_by_name(keyboard, "Numpad")

blender_utils.hide_collection(keyboard, False)
blender_utils.hide_collection(keyboard, SET_MASK, exclude=keys)

orientation.hide_set(True)
orientation.hide_render = True


def get_ith_color(i):
    r = K * i
    g = K * (r // 255)
    b = K * (g // 255)

    return r % 255, g % 255, b % 255


def create_mask_material(i):
    color = get_ith_color(i)
    name = f"ImageMask - {color}"

    if (mat := bpy.data.materials.get(name)) is None:
        mat = bpy.data.materials.new(name=name)

    # Set Material's Emission

    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes

    for node in nodes:
        if node.bl_idname not in {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial"}:
            nodes.remove(node)

    color = map(lambda c: c / 255, color)

    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color, 1)
    emit.inputs["Color"].keyframe_insert("default_value", frame=1)
    new_link = tree.links.new(nodes['Material Output'].inputs['Surface'], nodes['Emission'].outputs['Emission'])

    return mat


black = create_mask_material(0)

orientation = blender_utils.get_plane_normal(orientation)

for i, ob in enumerate(keys):
    assert i <= 255
    print(ob)

    mat = create_mask_material(i + 1)

    blender_utils.set_obj_material(ob, 1, black)
    blender_utils.set_obj_material(ob, 2, mat)

    for face in ob.data.polygons:
        if not SET_MASK:
            face.material_index = 0
            continue

        if abs(face.normal.dot(orientation) - 1) < 0.1:
            face.material_index = 2
        else:
            face.material_index = 1
