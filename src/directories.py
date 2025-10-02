import os
import sys

project_root = os.path.dirname(os.path.dirname(__file__))
python_dir = os.path.dirname(sys.executable)
environment_name = os.path.dirname(python_dir)


def abspath(p: str) -> str:
    import os

    p = os.path.expanduser(p)
    p = os.path.expandvars(p)

    return os.path.abspath(p)
