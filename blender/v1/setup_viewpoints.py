import bpy
import mathutils
import random


def point_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians.

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)

    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'), 'Z') if obj.type == 'CAMERA' else (
    ('X', 'Z'), 'Y')  # because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)

    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, rotator)

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    # obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc

    return direction


viewpoint_obj = bpy.data.objects['Viewpoints']
viewpoints = viewpoint_obj.data.vertices

keyboard = bpy.data.objects["Keyboard"]
camera = bpy.data.objects['Camera']

scales = (0, 1)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(viewpoints) * len(scales)

random.seed(0)


def randomize(ob):
    x = random.random() - 0.5
    y = random.random() - 0.5
    rx = (random.random() - 0.5) / 4
    ry = (random.random() - 0.5) / 4
    rz = (random.random() - 0.5) / 4

    bpy.ops.transform.translate(value=(x, y, 0), orient_type="LOCAL")
    bpy.ops.transform.rotate(value=rx, orient_axis="X")
    bpy.ops.transform.rotate(value=ry, orient_axis="Y")
    bpy.ops.transform.rotate(value=rz, orient_axis="Z")


bpy.ops.object.select_all(action="DESELECT")
bpy.data.objects["Camera"].select_set(True)

i = 0
for scale in scales:
    for point in viewpoints:
        camera.location = viewpoint_obj.matrix_world @ point.co
        direction = point_at(camera, keyboard.location, roll=0)
        camera.location -= scale * direction

        randomize(camera)
        camera.keyframe_insert(data_path="location", frame=i)
        camera.keyframe_insert(data_path="rotation_euler", frame=i)
        i += 1

        randomize(camera)
        camera.keyframe_insert(data_path="location", frame=i)
        camera.keyframe_insert(data_path="rotation_euler", frame=i)
        i += 1
