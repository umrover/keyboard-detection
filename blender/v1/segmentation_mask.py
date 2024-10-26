import bpy

SET_MASK = False


def show_object(obj, val=True, exclude=()):
    if obj in exclude:
        return

    obj.hide_set(not val)
    obj.hide_render = not val

    for child in obj.children:
        show_object(child, val, exclude)


keys = list(bpy.data.collections["Keys"].all_objects)
keyboard = bpy.data.objects["Aluminium"]
show_object(keyboard, not SET_MASK, exclude=keys)

orientation = bpy.data.objects["Orientation"]
show_object(orientation, False)


def configure_material(name, color):

    # Get/Create Material

    mat = bpy.data.materials.get(name)

    if mat is None:
        mat = bpy.data.materials.new(name=name)

    # Set Material's Emission

    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes

    # clear all nodes
    for node in nodes:
        if node.bl_idname not in {"ShaderNodeBsdfPrincipled", "ShaderNodeOutputMaterial"}:
            nodes.remove(node)

    color = map(lambda c: c / 255, color)

    # create new emission node
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color, 1)
    emit.inputs["Color"].keyframe_insert("default_value", frame=1)
    new_link = tree.links.new(nodes['Material Output'].inputs['Surface'], nodes['Emission'].outputs['Emission'])

    return mat


def set_obj_material(obj, index, mat):
    if len(obj.data.materials) == index:
        obj.data.materials.append(mat)

    obj.data.materials[index] = mat


black = configure_material("ImageMask - Black", (0, 0, 0))
orientation = orientation.data.polygons[0].normal

# Set Keys Material
for i, ob in enumerate(keys):
    assert i <= 255

    mat = configure_material(f"ImageMask - #{i + 1}", (i + 1, i + 1, i + 1))

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

# Set Rendering Config
if SET_MASK:
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
else:
    bpy.context.scene.render.engine = "CYCLES"
