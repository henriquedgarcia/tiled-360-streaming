{
    "project": "nasrabadi_28videos_erp_cmp",
    "dataset_name": "nasrabadi_28videos",
    "error_metric": "rmse",
    "decoding_num": 5,
    "fov": "110x90",
    "codec": "x265",
    "codec_params": "keyint=30:min-keyint=30:open-gop=0:scenecut=0:info=0",
    "fps": 30,
    "gop": 30,
    "plot_config": "graphs_nas_erp_cmp.json",
    "distributions": [
        "burr12",
        "fatiguelife",
        "gamma",
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
        "46",
        "40",
        "34",
        "28",
        "22",
        "16",
        "0"
    ],
    "tiling_list": [
        "12x8",
        "9x6",
        "6x4",
        "3x2",
        "1x1"
    ],
}
