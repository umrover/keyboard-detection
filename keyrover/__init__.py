from .paths import *
from .plotting import *
from .image import *

from .blender import *
from .util import *

import os
if os.getcwd().endswith("notebooks"):
    os.chdir(f"{os.path.dirname(os.path.realpath(__file__))}/..")
