from .paths import *

from .blender import *
from .util import *

from .plotting import *

import os
if os.getcwd().endswith("notebooks") or os.getcwd().endswith("scripts"):
    os.chdir(f"{os.path.dirname(os.path.realpath(__file__))}/..")
