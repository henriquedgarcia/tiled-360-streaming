{
    "project": "nasrabadi_28videos_erp_cmp",
    "dataset_name": "nasrabadi_28videos",
    "error_metric": "rmse",
    "decoding_num": 5,
    "fov": "110x90",
    "codec": "x265",
    "codec_params": "keyint=30:min-keyint=30:open-gop=0:scenecut=0:info=0",
    "scale": {'erp': "4320x2160",
              'cmp': "3240x2160"},
    "duration": 60,
    "fps": 30,
    "gop": 30,
    "distributions": [
        "burr12",
        "fatiguelife",
        "gamma",
        "beta",
        "invgauss",
        "rayleigh",
        "lognorm",
        "genpareto",
        "pareto",
        "halfnorm",
        "expon"
    ],
    "rate_control": "crf",
    "original_quality": "0",
    "quality_list": [
        "0",
        "16",
        "22",
        "28",
        "34",
        "40",
        "46"
    ],
    "tiling_list": [
        "1x1",
        "3x2",
        "6x4",
        "9x6",
        "12x8"
    ],
}


