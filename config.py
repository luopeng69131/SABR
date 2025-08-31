# =========================
# ABRBench CONFIG (paths match your latest folders)
# =========================

# Choose one:
# 'FCC-16','FCC-18','Oboe','Puffer-21','Puffer-22','HSR',
# 'Norway3G','Lumos4G','Lumos5G','SolisWi-Fi','Ghent','Lab',
# 'ABRBench-3G','ABRBench-4G+'
_DATASET = 'ABRBench-4G+'   # change as needed

# Base directories
BASE_DIR = './ABRBench'
VIDEO_DIR = f'{BASE_DIR}/video'
TRACE_DIR = f'{BASE_DIR}/trace'

_DATASET_OPTION = {
    # ---------- ABRBench-3G (use envivio_3g) ----------
    'FCC-16': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/FCC-16/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-3G/FCC-16/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/FCC-16/',
    },
    'FCC-18': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/FCC-18/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-3G/FCC-18/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/FCC-18/',
    },
    'Oboe': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/Oboe/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-3G/Oboe/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/Oboe/',
    },
    'Puffer-21': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/Puffer-21/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-3G/Puffer-21/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/Puffer-21/',
    },
    'Puffer-22': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/Puffer-22/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-3G/Puffer-22/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/Puffer-22/',
    },

    # ---------- OOD in 3G ----------
    'HSR': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-3G/HSR',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/HSR/',
    },

    # ---------- ABRBench-4G+ (use big_buck_bunny) ----------
    'Norway3G': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Norway3G/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Norway3G/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/Norway3G/',
    },
    'Lumos4G': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Lumos4G/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Lumos4G/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/Lumos4G/',
    },
    'Lumos5G': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Lumos5G/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Lumos5G/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/Lumos5G/',
    },
    'SolisWi-Fi': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/SolisWi-Fi/test',
        'TRAIN_TRACES': f'{TRACE_DIR}/ABRBench-4G+/SolisWi-Fi/train',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/SolisWi-Fi/',
    },

    # ---------- OOD in 4G+ ----------
    'Ghent': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Ghent',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/Ghent/',
    },
    'Lab': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': f'{TRACE_DIR}/ABRBench-4G+/Lab',
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/Lab/',
    },

    # ---------- Mixed (composite) ----------
    'ABRBench-3G': {
        'VIDEO_BIT_RATE': [300, 750, 1200, 1850, 2850, 4300],
        'REBUF_PENALTY': 4.3,
        'TEST_TRACES': [
            f'{TRACE_DIR}/ABRBench-3G/FCC-16/test',
            f'{TRACE_DIR}/ABRBench-3G/FCC-18/test',
            f'{TRACE_DIR}/ABRBench-3G/Oboe/test',
            f'{TRACE_DIR}/ABRBench-3G/Puffer-21/test',
            f'{TRACE_DIR}/ABRBench-3G/Puffer-22/test',
            f'{TRACE_DIR}/ABRBench-3G/HSR',
        ],
        'TRAIN_TRACES': [
            f'{TRACE_DIR}/ABRBench-3G/FCC-16/train',
            f'{TRACE_DIR}/ABRBench-3G/FCC-18/train',
            f'{TRACE_DIR}/ABRBench-3G/Oboe/train',
            f'{TRACE_DIR}/ABRBench-3G/Puffer-21/train',
            f'{TRACE_DIR}/ABRBench-3G/Puffer-22/train',
        ],
        'FINE_TUNE_TRACES': [],
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/envivio_3g/video_size_',
        'LOG_FILE_DIR': './test_results/3g_mix/',
    },

    'ABRBench-4G+': {
        'VIDEO_BIT_RATE': [1000, 2500, 5000, 8000, 16000, 40000],
        'REBUF_PENALTY': 40,
        'TEST_TRACES': [
            f'{TRACE_DIR}/ABRBench-4G+/Norway3G/test',
            f'{TRACE_DIR}/ABRBench-4G+/Lumos4G/test',
            f'{TRACE_DIR}/ABRBench-4G+/Lumos5G/test',
            f'{TRACE_DIR}/ABRBench-4G+/SolisWi-Fi/test',
            f'{TRACE_DIR}/ABRBench-4G+/Ghent',
            f'{TRACE_DIR}/ABRBench-4G+/Lab',
        ],
        'TRAIN_TRACES': [
            f'{TRACE_DIR}/ABRBench-4G+/Norway3G/train',
            f'{TRACE_DIR}/ABRBench-4G+/Lumos4G/train',
            f'{TRACE_DIR}/ABRBench-4G+/Lumos5G/train',
            f'{TRACE_DIR}/ABRBench-4G+/SolisWi-Fi/train',
        ],
        'FINE_TUNE_TRACES': [],
        'VIDEO_SIZE_FILE': f'{VIDEO_DIR}/big_buck_bunny/video_size_',
        'LOG_FILE_DIR': './test_results/4gplus_mix/',
    },
}

# ------------- auto load & checks -------------
config = _DATASET_OPTION[_DATASET]

VIDEO_BIT_RATE = config['VIDEO_BIT_RATE']
REBUF_PENALTY = config['REBUF_PENALTY']
TEST_TRACES = config['TEST_TRACES']
TRAIN_TRACES = config.get('TRAIN_TRACES', None)
VIDEO_SIZE_FILE = config['VIDEO_SIZE_FILE']
LOG_FILE_DIR = config['LOG_FILE_DIR']

DATASET_NAME = _DATASET

import os
if not os.path.exists(LOG_FILE_DIR):
    os.makedirs(LOG_FILE_DIR)
    print(f"LOG_FILE_DIR '{LOG_FILE_DIR}' created.")

def assert_paths_exist(paths, name):
    if isinstance(paths, str):
        assert os.path.exists(paths), f"{name} path not exist: {paths}"
    elif isinstance(paths, list):
        for p in paths:
            assert os.path.exists(p), f"{name} path not exist: {p}"
    else:
        raise TypeError(f"{name} should be list or str, got: {type(paths)}")

if TRAIN_TRACES is not None:
    assert_paths_exist(TRAIN_TRACES, "training data")
assert_paths_exist(TEST_TRACES, "test data")
