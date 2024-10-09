from typing import NamedTuple

Option = NamedTuple('Option', [('id', int), ('name', str), ('obj', object)])


def make_help_txt(config, video, worker):
    config_options = show_options(config, 1, True)
    video_options = show_options(video, 1, True)
    worker_options = show_options(worker, 1, True)
    text = (f'Dectime Testbed.\n'
            f'================\n'
            f'CONFIG_ID:\n'
            f'{config_options}\n'
            f'VIDEOS_LIST_ID\n'
            f'{video_options}\n'
            f'WORKER_ID\n'
            f'{worker_options}')
    return text


def show_options(options_list: list[Option], indent=0, silent=False):
    txt = "\n".join(['\t' * indent + f'{o.id} - {o.name}' for o in options_list])
    if not silent:
        print(txt)
    return txt


def request_options(options_list) -> Option:
    chosen = int(input(f'Option: '))
    options = get_option(chosen, options_list)
    return options


def get_option(opt_id: int, options_list) -> Option:
    for opt in options_list:
        if opt_id == opt.id:
            return opt
    raise ValueError


def menu(options_list, init_indent=0):
    while True:
        show_options(options_list, indent=init_indent)
        try:
            option = request_options(options_list)
            return option
        except ValueError:
            continue
