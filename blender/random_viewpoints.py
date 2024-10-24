import bpy
from mathutils import Vector, Matrix
from random import uniform
from numpy import random
from math import radians, degrees, cos, acos, sin

# PARAMETERS CONFIGURATION
# --------------------------------------------


theta_lims = (-40, 20)  # degrees (0-360), x-axis
theta_std = 0.3

phi_lims = (65, 115)  # degrees (0-180), y-axis
phi_std = 0.2

rho_lims = (2, 5)
max_roll = 15

N = 1000

LOCATION_SAMPLING = "gaussian"
THETA_SAMPLING = "gaussian"
PHI_SAMPLING = "gaussian"

# SCRIPT BEGINS HERE
# --------------------------------------------


keyboard_plane = bpy.data.objects["Orientation"]
camera = bpy.data.objects["Camera"]


def get_random_arc_angle(theta1, theta2, phi1, phi2):
    theta1 = radians(theta1)
    theta2 = radians(theta2)
    phi1 = radians(phi1)
    phi2 = radians(phi2)

    if THETA_SAMPLING == "uniform":
        theta = uniform(theta1, theta2)

    elif THETA_SAMPLING == "gaussian":
        theta = random.normal((theta1 + theta2) / 2, theta_std)
        # theta = max(theta1, min(theta2, theta))

    else:
        raise ValueError("Unknown Theta Sampling")

    # x1 = (1 - cos(phi1)) / 2
    # x2 = (1 - cos(phi2)) / 2
    # x = uniform(x1, x2)
    # phi = arccos(1 - 2 * x)

    # The above code simplifies to

    x1 = cos(phi1)
    x2 = cos(phi2)

    if PHI_SAMPLING == "uniform":
        x = uniform(x1, x2)

    elif PHI_SAMPLING == "gaussian":
        x = random.normal((x1 + x2) / 2, phi_std)
        x = max(-1, min(1, x))

    else:
        raise ValueError("Unknown Phi Sampling")

    phi = acos(x)

    return theta, phi


def spherical_to_cartesian(rho, theta, phi):
    x = rho * sin(phi) * cos(theta)
    y = rho * sin(phi) * sin(theta)
    z = rho * cos(phi)

    return z, y, x


def point_at(camera, target, roll=0):
    direction = target - camera.location
    tracker, rotator = (("-Z", "Y"), "Z")
    quat = direction.to_track_quat(*tracker)

    quat = quat.to_matrix().to_4x4()
    roll = radians(roll)
    roll_matrix = Matrix.Rotation(roll, 4, rotator)

    loc = camera.location.to_tuple()
    camera.matrix_world = quat @ roll_matrix
    camera.location = loc

    return direction


def delete_empty_objects():
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.name.startswith("Empty"))
    bpy.ops.object.delete()


delete_empty_objects()

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = N

for i in range(N):
    random_angle = get_random_arc_angle(*theta_lims, *phi_lims)
    random_dist = uniform(*rho_lims)
    random_vec = spherical_to_cartesian(random_dist, *random_angle)
    random_vec = Vector(random_vec)

    if LOCATION_SAMPLING == "gaussian":
        random_x = max(-1, min(1, random.normal(0, 0.5)))
        random_y = max(-1, min(1, random.normal(0, 0.5)))

    elif LOCATION_SAMPLING == "uniform":
        random_x = uniform(-1, 1)
        random_y = uniform(-1, 1)

    else:
        raise ValueError("Unknown Location Sampling")

    random_translation = Vector((random_x, random_y, 0))
    random_translation = keyboard_plane.matrix_world @ random_translation
    random_vec += random_translation

    random_roll = uniform(-max_roll, max_roll)

    # Create Camera Keyframe

    camera.location = random_vec
    point_at(camera, random_translation, random_roll)

    camera.keyframe_insert(data_path="location", frame=i)
    camera.keyframe_insert(data_path="rotation_euler", frame=i)

    if uniform(0, 1) > 100 / N:
        continue

    # Add line showing angle

    mesh = bpy.data.meshes.new("Empty")
    obj = bpy.data.objects.new("Empty", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata([random_translation, random_vec], [(0, 1)], [])
    mesh.update()

    # bpy.ops.object.empty_add(type="SPHERE", radius=0.05, align="WORLD", location=random_vec, scale=(1, 1, 1))
