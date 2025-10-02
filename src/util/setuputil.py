from typing import Callable, Optional


def yesno(prompt: str, default: bool = True) -> bool:
    while True:
        x = input(f'{prompt} ({"YES" if default else "yes"}/{"NO" if not default else "no"})')

        x = x.lower()

        if x == '':
            return default
        elif x in ('y', 'yes'):
            return True
        elif x in ('n', 'no'):
            return False
        elif x == 'exit':
            raise KeyboardInterrupt()


def input_sanitized(prompt: str,
                    sanitize: Callable[[str], bool],
                    transform: Optional[Callable[[str], str]] = None,
                    default_value: Optional[str] = None,
                    commit_action: Optional[Callable[[str], bool]] = None, ) -> str:
    if transform is not None and default_value is not None:
        default_value = transform(default_value)

    if default_value is not None and not sanitize(default_value):
        raise ValueError('`default_value` must pass `sanitize`, but `sanitize` returned `False`.')

    while True:
        while True:
            q = input(prompt)

            if q == '':
                if default_value is None:
                    continue

                q = default_value
                break

            if transform is not None:
                q = transform(q)

            if sanitize(q):
                break

        # if not yesno(f'Setting to value\n    {q}\nOk?'):
        #     continue

        if commit_action is None or commit_action(q):
            break

    return q


def press_enter_to_exit(raise_keyboard_interrupt: bool = True) -> None:
    input('Press enter to exit...')
    if raise_keyboard_interrupt:
        raise KeyboardInterrupt()

