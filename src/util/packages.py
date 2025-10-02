from typing import Union


def install(package: Union[str, dict[str, str]]) -> None:
    import subprocess
    import sys
    import os

    python_executable = sys.executable

    args = []

    try:
        import pip
        s = 'pip'
    except ModuleNotFoundError:
        raise RuntimeError('Unsupported package manager.')

    args.append(os.path.join(os.path.dirname(sys.executable), s))
    args.append('install')

    if isinstance(package, dict):
        args.append(package[s])
    else:
        args.append(package)

    subprocess.run(args, cwd=os.path.dirname(python_executable))
