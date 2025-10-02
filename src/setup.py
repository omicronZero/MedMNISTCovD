from subprocess import CompletedProcess

from util.setuputil import yesno, press_enter_to_exit, input_sanitized
from directories import abspath, python_dir, project_root
import sys
import os
from typing import Any, overload, Literal, Union
import config

project_name = 'MedMNISTCovD'

torch_version = (2, 2, 1)
torchvision_version = (0, 17, 1)


@overload
def run(name: str, *args: Any, check: bool = True, reruns: int = 0, text: Literal[True] = False) \
        -> CompletedProcess[str]:
    ...


@overload
def run(name: str, *args: Any, check: bool = True, reruns: int = 0, text: Literal[False] = False) \
        -> CompletedProcess[bytes]:
    ...


@overload
def run(name: str, *args: Any, check: bool = True, reruns: int = 0, text: bool = False) \
        -> CompletedProcess[Union[str, bytes]]:
    ...


def run(name: str, *args: Any, check: bool = True, reruns: int = 0, text: bool = False) -> CompletedProcess[str]:
    import subprocess
    while True:
        try:
            return subprocess.run([os.path.join(python_dir, name), *args], check=check, text=text)
        except subprocess.CalledProcessError:
            if reruns > 0:
                reruns -= 1
            else:
                raise


def try_make_dir(dir: str) -> bool:
    dir = abspath(dir)
    try:
        os.makedirs(dir, exist_ok=True)
        return True
    except BaseException as ex:
        print(ex)
        return False


def setup() -> None:
    major, minor, *_ = sys.version_info

    if major != 3 or minor < 11:
        print('Python version must be 3.11 or later.')
        return press_enter_to_exit()

    used_dirs = []

    def sanitize_dir(d: str) -> bool:
        if d in used_dirs:
            print('The indicated directory is already being used in one of the other modes. Please specify another '
                  'target.')
            return False

        return True

    def commit_dir(d: str) -> bool:
        try:
            os.makedirs(d)
        except FileExistsError:
            if not yesno('The indicated directory exists already. Use nonetheless?'):
                return False
        except BaseException as ex:
            print('Failed to create the directory due to the following error:')

            import warnings
            warnings.warn(str(ex))

            return False

        used_dirs.append(d)
        return True

    cfg = config.load_user_config()

    if cfg is None:
        print('Please configure the locations in which the respective data will be stored:')

        dataset_dir = input_sanitized(
            'Dataset directory:',
            sanitize_dir,
            transform=abspath,
            default_value=f'~/{project_name}/data',
            commit_action=commit_dir
        )

        cache_dir = input_sanitized(
            'Cache directory:',
            sanitize_dir,
            transform=abspath,
            default_value=f'~/{project_name}/cache',
            commit_action=commit_dir
        )

        model_dir = input_sanitized(
            'Model directory:',
            sanitize_dir,
            transform=abspath,
            default_value=f'~/{project_name}/models',
            commit_action=commit_dir
        )

        result_dir = input_sanitized(
            'Result directory:',
            sanitize_dir,
            transform=abspath,
            default_value=f'~/{project_name}/results',
            commit_action=commit_dir
        )

        print('Creating local configuration file using the supplied settings:')
        print(f'    {config.get_default_config_path()}')

        config.save_user_config(config.Config(cache_dir, dataset_dir, result_dir, model_dir))

    print('Sanitizing environment...')

    try:
        import torch

        version = torch.__version__

        major, minor, rev = map(int, version.split('+')[0].split('.'))

        if (major, minor, rev) != torch_version:
            if not yesno(
                    'Existing torch installation found in environment, but the version is different to the expected '
                    f'version 2.5.1 (found: {version}). Continue?'):
                print('Please select a new environment or uninstall the currently installed version manually.')
                return press_enter_to_exit()
    except ModuleNotFoundError:
        torch = None

    print()

    print(
        'Please make sure that the following path is the one you want to install the required dependencies to. Be '
        'sure that it is a Python environment. The packages required by the application will be installed to this '
        'directory:')
    print('    ', python_dir)

    if not yesno('Is this path the path of the environment you want to use?'):
        print('Please restart the setup in the environment you want to use.')
        return press_enter_to_exit()

    import platform
    operating_system = platform.system()

    if torch is None:
        print('Downloading and installing PyTorch... Please be patient, this will take time.')

        if operating_system in ('Windows', 'Linux'):
            run('pip', 'install', '--no-dependencies',
                f'torch=={".".join(map(str, torch_version))}',
                f'torchvision=={".".join(map(str, torchvision_version))}',
                '--index-url', 'https://download.pytorch.org/whl/cu121')
        elif operating_system == 'Darwin':
            if not yesno('CUDA is unavailable on Mac. Should the default version be installed?'):
                print('Please install PyTorch manually or use another operating system.')
                return press_enter_to_exit()

            run('pip', 'install', '--no-dependencies', 'torch', 'torchvision')
        else:
            print(
                f'Unsupported platform: {operating_system} {platform.release()}. Please install PyTorch manually '
                f'before continuing.')
            return press_enter_to_exit()

    requirements_txt = os.path.join(project_root, "requirements.txt")

    print('Downloading and installing the required project packages from \'requirements.txt\'... '
          'This may take a while.')

    run('pip', 'install', '-r', requirements_txt, reruns=1)

    print('Done installing requirements.')

    print('Sanitizing PyTorch installation. This may take a moment.')

    import torch

    if not torch.cuda.is_available() and operating_system in ('Windows', 'Linux'):
        print('CUDA is unavailable. If not intended, make sure that CUDA-version 12.1 is supported.')

        if yesno('Query info about your CUDA installation?'):
            run('nvidia-smi', check=False)

        if not yesno('Continue?'):
            return press_enter_to_exit()

    print('Checking for `segment_anything`-package via medsam...')

    try:
        import segment_anything
        print(
            'Existing installation of `segment_anything` package found. This installation will be used by the MedSAM-'
            'based functionality.')
    except ModuleNotFoundError:
        print('`segment_anything` not found. Installing from the `medsam`-repository. This will take a while.')

        install_medsam()

        print('Done installing \'medsam\' package.')


def install_medsam() -> None:
    from util.packages import install

    install('git+https://github.com/bowang-lab/MedSAM.git@a7b77769ff12035414d0aaf3bc87230b7c10f922')


if __name__ == '__main__':
    print('Running setup for the application.')
    try:
        setup()
        print('We\'re done with the setup.')
        press_enter_to_exit()
    except KeyboardInterrupt:
        pass
