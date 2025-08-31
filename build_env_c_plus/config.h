#ifndef CONFIG_H
#define CONFIG_H

#include <string>

/*
Dataset options (set DATASET_OPTION below):
  1  : FCC-16
  2  : FCC-18
  3  : Oboe
  4  : Puffer-21
  5  : Puffer-22
  6  : Norway3G
  7  : Lumos4G
  8  : Lumos5G
  9  : SolisWi-Fi
  15 : HSR           (OOD)
  16 : Ghent         (OOD)
  17 : Lab           (OOD)
  20 : ABRBench-3G   (mix; leave TEST_TRACES empty, handled in Python)
  30 : ABRBench-4G+  (mix; leave TEST_TRACES empty, handled in Python)
*/

// !!! change here to choose dataset
#define DATASET_OPTION 30

// Only used by DP runner (single-dir evaluation).
// Mixed sets (20/30) are left empty on purpose and assembled on the Python side.

// Common base dirs
static const std::string BASE_DIR  = "./ABRBench";
static const std::string VIDEO_DIR = BASE_DIR + "/video";
static const std::string TRACE_DIR = BASE_DIR + "/trace";

// -------------------------- 3G trace sets (envivio_3g) --------------------------
#if DATASET_OPTION == 1
// FCC-16
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/FCC-16/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/FCC-16/";

#elif DATASET_OPTION == 2
// FCC-18
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/FCC-18/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/FCC-18/";

#elif DATASET_OPTION == 3
// Oboe
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/Oboe/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Oboe/";

#elif DATASET_OPTION == 4
// Puffer-21
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/Puffer-21/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Puffer-21/";

#elif DATASET_OPTION == 5
// Puffer-22
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/Puffer-22/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Puffer-22/";

// -------------------------- OOD in 3G --------------------------
#elif DATASET_OPTION == 15
// HSR (OOD, raw traces)
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-3G/HSR/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/HSR/";

// -------------------------- 4G+ trace sets (big_buck_bunny) --------------------------
#elif DATASET_OPTION == 6
// Norway3G
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/Norway3G/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Norway3G/";

#elif DATASET_OPTION == 7
// Lumos4G
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/Lumos4G/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Lumos4G/";

#elif DATASET_OPTION == 8
// Lumos5G
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/Lumos5G/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Lumos5G/";

#elif DATASET_OPTION == 9
// SolisWi-Fi  (注意：目录名为 "SolisWi-Fi")
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/SolisWi-Fi/test/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/SolisWi-Fi/";

// -------------------------- OOD in 4G+ --------------------------
#elif DATASET_OPTION == 16
// Ghent (OOD, raw traces)
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/Ghent/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Ghent/";

#elif DATASET_OPTION == 17
// Lab (OOD, raw traces)
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = TRACE_DIR + "/ABRBench-4G+/Lab/";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "./test_results/Lab/";

// -------------------------- Mixed (assembled elsewhere) --------------------------
#elif DATASET_OPTION == 20
// ABRBench-3G (mix) — leave empty; DP side assembles a list of folders
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {300, 750, 1200, 1850, 2850, 4300};
const double REBUF_PENALTY = 4.3;
const std::string TEST_TRACES     = "";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/envivio_3g/video_size_";
const std::string LOG_FILE_DIR    = "";

#elif DATASET_OPTION == 30
// ABRBench-4G+ (mix) — leave empty; DP side assembles a list of folders
const int    BITRATE_LEVELS = 6;
const double VIDEO_BIT_RATE[BITRATE_LEVELS] = {1000, 2500, 5000, 8000, 16000, 40000};
const double REBUF_PENALTY = 40.0;
const std::string TEST_TRACES     = "";
const std::string VIDEO_SIZE_FILE = VIDEO_DIR + "/big_buck_bunny/video_size_";
const std::string LOG_FILE_DIR    = "";

#else
#error "Invalid DATASET_OPTION value!"
#endif

#endif // CONFIG_H
