from .paths import *
from .plotting import *

from .blender import *
from .util import *

import os
if os.getcwd().endswith("notebooks") or os.getcwd().endswith("scripts"):
    os.chdir(f"{os.path.dirname(os.path.realpath(__file__))}/..")
