import bpy
from mathutils import Vector

# PARAMETERS CONFIGURATION
# --------------------------------------------


SET_MASK = False

keyboard = 1

TOTAL_KEYBOARDS = 4


# SCRIPT BEGINS HERE
# --------------------------------------------


# See https://blender.stackexchange.com/questions/169672/object-hide-render-crashes-blender-before-starting-to-bake
def get_collection_objects(collection):
    obj_names = [obj.name for obj in collection.all_objects]

    for name in obj_names:
        obj = bpy.data.objects.get(name)
        yield obj


def hide_collection(collection, val=True, exclude=()):
    for obj in get_collection_objects(collection):
        if obj in exclude:
            continue

        obj.hide_render = val
        obj.hide_set(val)


def get_children_by_name(collection, name, all=True):
    objs = []

    for child in collection.all_objects:
        if child.name.startswith(name):
            if child.data is None:
                continue

            if all:
                objs.append(child)
                continue

            return child

    return objs


# See https://blender.stackexchange.com/questions/27491/python-vertex-normal-according-to-world
def get_object_normal(obj):
    normal = obj.data.vertices[0].normal.to_4d()
    normal.w = 0
    normal = (obj.matrix_world @ normal).to_3d()
    return normal / normal.length


for i in range(TOTAL_KEYBOARDS):
    objs = bpy.data.collections[f"Keyboard {i + 1}"]
    hide_collection(objs, True)

keyboard = bpy.data.collections[f"Keyboard {keyboard}"]
orientation = get_children_by_name(keyboard, "Orientation", all=False)
keys = get_children_by_name(keyboard, "Key")
keys += get_children_by_name(keyboard, "Numpad")

hide_collection(keyboard, False)
hide_collection(keyboard, SET_MASK, exclude=keys)

orientation.hide_set(True)
orientation.hide_render = True

k = 2


def get_ith_color(i):
    r = i * k
    g = k * (r // 255)
    b = k * (g // 255)

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


def set_obj_material(obj, index, mat):
    if len(obj.data.materials) == index:
        obj.data.materials.append(mat)

    obj.data.materials[index] = mat


black = create_mask_material(0)

z = Vector((0, 0, 1))
orientation = get_object_normal(orientation)
print(orientation)

# Set Keys Material
for i, ob in enumerate(keys):
    assert i <= 255
    print(ob)

    mat = create_mask_material(i + 1)

    set_obj_material(ob, 1, black)
    set_obj_material(ob, 2, mat)

    for face in ob.data.polygons:
        if not SET_MASK:
            face.material_index = 0
            continue

        if abs(face.normal.dot(orientation) - 1) < 0.1:
            face.material_index = 2
        else:
            face.material_index = 1
