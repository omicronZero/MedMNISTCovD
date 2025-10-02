import os.path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    cache_dir: str
    dataset_dir: str
    result_dir: str
    model_dir: str
    

default_config_file: str = 'config.json'
default_indent: int = 2


def user_id() -> str:
    import socket
    import hashlib

    host_name = hashlib.sha1(socket.gethostname().encode('utf-8')).hexdigest()

    return host_name


def get_default_config_path() -> str:
    path = default_config_file

    if not os.path.isabs(path):
        from directories import project_path
        path = project_path(path)

    return path


def load_configs(path: str = ...) -> dict[str, Config]:
    if path is ...:
        path = get_default_config_path()

    try:
        with open(path, 'r', encoding='utf-8') as stream:
            import json
            objs = json.load(stream)
    except FileNotFoundError:
        return {}

    if not isinstance(objs, dict):
        raise RuntimeError('Malformed file \'config.json\': Expected a dictionary at the root.')

    return {user: Config(**obj) for user, obj in objs.items()}


def save_config(config: dict[str, Config], path: str = ..., update: bool = True) -> None:
    if update:
        if path is ...:
            path = get_default_config_path()

        try:
            with open(path, 'r+', encoding='utf-8') as stream:
                import json
                objs = json.load(stream)

                if not isinstance(objs, dict):
                    raise RuntimeError('Malformed file \'config.json\': Expected a dictionary at the root.')

                objs.update({user: asdict(cfg) for user, cfg in config.items()})

                if stream.truncate(0) == 0:
                    stream.seek(0)
                    json.dump(objs, stream, indent=default_indent)
                    return
        except FileNotFoundError:
            pass

    # we didn't update or the update failed
    with open(path, 'w', encoding='utf-8') as stream:
        import json
        json.dump({user: asdict(cfg) for user, cfg in config.items()}, stream, indent=default_indent)


def load_user_config(path: str = ...) -> Optional[Config]:
    return load_configs(path).get(user_id())


def save_user_config(config: Config, path: str = ...) -> None:
    save_config({user_id(): config}, path)
