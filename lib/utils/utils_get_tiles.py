from collections import Counter
from time import time

import numpy as np
from matplotlib import pyplot as plt
from py360tools import ERP, CMP

from config.config import config
from lib.assets.autodict import AutoDict
from lib.assets.context import ctx
from lib.assets.logger import logger
from lib.assets.paths.gettilespaths import get_tiles_paths
from lib.assets.paths.segmenterpaths import segmenter_paths
from lib.utils.util import load_json, save_json, splitx, print_error


class HMDDatasetError(Exception): ...

def start_get_tiles():
    init()
    for_each_video()


def init():
    ctx.hmd_dataset = load_json(get_tiles_paths.dataset_json)
    ctx.tiling_list.remove('1x1')
    ctx.tiles_1x1 = {'frame': [["0"]] * config.n_frames,
                     'chunks': {str(i): ["0"] for i in range(1, int(config.duration) + 1)}}

    ctx.projection_dict = AutoDict()
    for tiling in ctx.tiling_list:
        erp = ERP(tiling=tiling, proj_res='1080x540', vp_res='660x540', fov_res=config.fov)
        cmp = CMP(tiling=tiling, proj_res='810x540', vp_res='660x540', fov_res=config.fov)
        ctx.projection_dict['erp'][tiling] = erp
        ctx.projection_dict['cmp'][tiling] = cmp


def for_each_video():
    for ctx.name in ctx.name_list:
        print(f'==== GetTiles {ctx.name} ====')

        if logger.get_status('get_tiles_ok'):
            print(f'\t{ctx.name} is OK. Skipping')
            continue

        ctx.changed_flag = False
        ctx.error_flag = False
        load_results()

        try:
            for_each_projection()

        finally:
            if ctx.changed_flag:
                print('\n\tSaving.')
                save_results()
            if not ctx.error_flag:
                ctx.projection = ctx.tiling = ctx.tile = ctx.user = None
                logger.update_status('get_tiles_ok', True)


def for_each_projection():
    for ctx.projection in ctx.projection_list:
        for_each_tiling()


def for_each_tiling():
    for ctx.tiling in ctx.tiling_list:
        for_each_user()


def for_each_user():
    for ctx.user in ctx.user_list:
        print(f'==== GetTiles {ctx} ====')
        try:
            get_tiles_by_video()
        except (HMDDatasetError,):

        # self.count_tiles()
        # self.heatmap()
        # self.plot_test()


def get_tiles_by_video():
    start = time()

    user_hmd_data = get_user_hmd_data()
    if user_hmd_data == {}:
        logger.register_log(f'HMD samples is missing, user{ctx.user}')
        raise HMDDatasetError(f'HMD samples is missing, user{ctx.user}')

    tiles_seen_by_frame = get_tiles_seen_by_frame(user_hmd_data)
    tiles_seen_by_chunk = get_tiles_seen_by_chunk(tiles_seen_by_frame)

    tiles_seen = {'frames': tiles_seen_by_frame,
                  'chunks': tiles_seen_by_chunk}

    save_json(tiles_seen, get_tiles_paths.user_tiles_seen_json)
    print(f'\ttime =  {time() - start}')


def get_tiles_seen_by_frame(user_hmd_data):
    if ctx.tiling == '1x1':
        return ctx.tiles_1x1['frame']

    tiles_seen_by_frame = []
    projection_obj = ctx.projection_dict[ctx.projection][ctx.tiling]

    for frame, yaw_pitch_roll in enumerate(user_hmd_data, 1):
        print(f'\r\tframe {frame:04d}/{config.n_frames}', end='')
        vptiles: list[str] = projection_obj.get_vptiles(yaw_pitch_roll)
        vptiles = list(map(str, map(int, vptiles)))
        tiles_seen_by_frame.append(vptiles)
    return tiles_seen_by_frame


def get_tiles_seen_by_chunk(tiles_seen_by_frame):
    if ctx.tiling == '1x1':
        return ctx.tiles_1x1['chunks']

    tiles_seen_by_chunk = {}
    tiles_in_chunk = set()
    for frame, vptiles in enumerate(tiles_seen_by_frame):
        tiles_in_chunk.update(vptiles)

        if (frame + 1) % 30 == 0:
            chunk_id = frame // 30 + 1  # chunk start from 1
            tiles_seen_by_chunk[f'{chunk_id}'] = list(tiles_in_chunk)
            tiles_in_chunk.clear()
    return tiles_seen_by_chunk





def get_user_hmd_data():
    user_hmd_data = ctx.hmd_dataset[ctx.name + '_nas'][ctx.user]
    if user_hmd_data == {}:
        print_error(f'\tHead movement user samples are missing.')
        logger.register_log(f'HMD samples is missing, user{ctx.user}')
        ctx.error_flag = True
        raise
    return ctx.hmd_dataset[ctx.name + '_nas'][ctx.user]


def load_results():
    try:
        ctx.results = load_json(segmenter_paths.get_tiles_json,
                                object_hook=AutoDict)
    except FileNotFoundError:
        ctx.results = AutoDict()


def save_results():
    segmenter_paths.get_tiles_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(ctx.results, segmenter_paths.get_tiles_json)
    pass


def heatmap():
    results = load_json(segmenter_paths.counter_tiles_json)

    for ctx.tiling in ctx.tiling_list:
        if ctx.tiling == '1x1': continue

        filename = f'heatmap_tiling_{ctx.hmd_dataset_name}_{ctx.projection}_{ctx.name}_{ctx.tiling}_fov{config.fov}.png'
        heatmap_tiling = (segmenter_paths.get_tiles_folder / filename)
        if heatmap_tiling.exists(): continue

        tiling_result = results[ctx.tiling]

        h, w = splitx(ctx.tiling)[::-1]
        grade = np.zeros((h * w,))

        for item in tiling_result: grade[int(item)] = tiling_result[item]
        grade = grade.reshape((h, w))

        fig, ax = plt.subplots()
        im = ax.imshow(grade, cmap='jet', )
        ax.set_title(f'Tiling {ctx.tiling}')
        fig.colorbar(im, ax=ax, label='chunk frequency')

        # fig.show()
        fig.savefig(f'{heatmap_tiling}')


def count_tiles():
    if segmenter_paths.counter_tiles_json.exists(): return

    ctx.results = load_json(segmenter_paths.get_tiles_json)
    result = {}

    for ctx.tiling in ctx.tiling_list:
        if ctx.tiling == '1x1': continue

        # <editor-fold desc="Count tiles">
        tiles_counter_chunks = Counter()  # Collect tiling count

        for ctx.user in ctx.users_list:
            result_chunks: dict[str, list[str]] = ctx.results[ctx.projection][ctx.name][ctx.tiling][ctx.user][
                'chunks']

            for chunk in result_chunks:
                tiles_counter_chunks = tiles_counter_chunks + Counter(result_chunks[chunk])
        # </editor-fold>

        print(tiles_counter_chunks)
        dict_tiles_counter_chunks = dict(tiles_counter_chunks)

        # <editor-fold desc="Normalize Counter">
        nb_chunks = sum(dict_tiles_counter_chunks.values())
        for ctx.tile in ctx.tile_list:
            try:
                dict_tiles_counter_chunks[ctx.tile] /= nb_chunks
            except KeyError:
                dict_tiles_counter_chunks[ctx.tile] = 0
        # </editor-fold>

        result[ctx.tiling] = dict_tiles_counter_chunks

    save_json(result, segmenter_paths.counter_tiles_json)
