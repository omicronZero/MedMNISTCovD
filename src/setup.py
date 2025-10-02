from util.setuputil import yesno, press_enter_to_exit, input_sanitized
from directories import abspath, python_dir, project_root
import sys
import os
from typing import Any
import config

project_name = 'MedMNISTCovD'


def run(name: str, *args: Any, check: bool = True, reruns: int = 0) -> None:
    import subprocess
    while True:
        try:
            subprocess.run([os.path.join(python_dir, name), *args], check=check)
            break
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

    print('Sanitizing environment...')

    try:
        import torch

        version = torch.__version__

        major, minor, rev = map(int, version.split('+')[0].split('.'))

        if (major, minor, rev) != (2, 5, 1):
            if not yesno(
                    'Existing torch installation found in environment, but the version is different to the expected '
                    f'version 2.5.1 (found: {version}). Continue?'):
                press_enter_to_exit()
                return

    except ModuleNotFoundError:
        torch = None

    print()

    print(
        'Please make sure that the following path is the one you want to install the required dependencies to. Be '
        'sure that it is a Python environment. The \'requirements.txt\' '
        'required by the application will be installed to this directory:')
    print('    ', python_dir)

    if not yesno('Is this path the path of the environment you want to use?'):
        print('Please restart the setup in the environment you want to use.')
        press_enter_to_exit()
        return

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

    import platform
    operating_system = platform.system()

    if torch is None:
        print('Downloading and installing PyTorch... Please be patient, this will take time.')

        if operating_system in ('Windows', 'Linux'):
            run('pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url',
                'https://download.pytorch.org/whl/cu121')
        elif operating_system == 'Darwin':
            if not yesno('CUDA is unavailable on Mac. Should the default version be installed?'):
                print('Please install PyTorch manually or use another operating system.')
                return press_enter_to_exit()

            run('pip', 'install', 'torch', 'torchvision', 'torchaudio')
        else:
            print(
                f'Unsupported platform: {operating_system} {platform.release()}. Please install PyTorch manually '
                f'before continuing.')
            return press_enter_to_exit()

    print('Sanitizing PyTorch installation. This may take a moment.')

    import torch

    if not torch.cuda.is_available() and operating_system in ('Windows', 'Linux'):
        print('CUDA is unavailable. If not intended, make sure that CUDA-version 12.1 is supported.')

        if yesno('Query info about your CUDA installation?'):
            run('nvidia-smi', check=False)

        if not yesno('Continue?'):
            return

    print('Downloading and installing the required project packages from \'requirements.txt\'... '
          'This may take a while.')

    run('pip', 'install', '-r', os.path.join(project_root, "requirements.txt"), reruns=1)

    print('Done installing requirements.')

    print('Checking for `segment_anything`-package via medsam...')

    try:
        import segment_anything
        print(
            'Existing installation of `segment_anything` package found. This installation will be used by the MedSAM-'
            'based functionality.')
    except ModuleNotFoundError:
        print('`segment_anything` not found. Installing from the `medsam`-repository. This will take a while.')

        from pretrained import install_medsam

        install_medsam()

        print('Done installing \'medsam\' package.')


if __name__ == '__main__':
    print('Running setup for the application.')
    try:
        setup()
        print('We\'re done with the setup.')
        press_enter_to_exit()
    except KeyboardInterrupt:
        pass
