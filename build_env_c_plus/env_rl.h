#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iterator>
#include <sstream>
#include <sys/types.h>
#include "dirent.h"
#include <errno.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
class Environment
{
public:
	//void start();
    Environment(std::vector<std::vector<double>> all_cooked_time, 
	    std::vector<std::vector<double>> all_cooked_bw, int seed = 42, double buffer_w = 0.0, bool is_train = true);
    ~Environment(){;};
	double get_download_time(int video_chunk_size);
    void reset_download_time();
	std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
        get_video_chunk(int quality, bool switch_trace);
	int get_optimal(int last_bit_rate, int top_k, int horizon);
	int optimal;
    bool is_train;
	// double delay0;
	// double sleep_time0;
	// double return_buffer_size0;
	// double rebuf0;
	// double video_chunk_size0;
	// bool end_of_video0;
	// double video_chunk_remain0;
	// double video_chunk_vmaf0;

    vector<vector<double>> all_cooked_bw;
    vector<vector<double>> all_cooked_time;
    vector<vector<int>> CHUNK_COMBO_OPTIONS;
    vector<string> all_file_names;
    int video_chunk_counter;
    double buffer_size;
    int trace_idx;
    vector<double> cooked_time;
    vector<double> cooked_bw;
    int mahimahi_start_ptr;
    int mahimahi_ptr;
    double last_mahimahi_time;
    int virtual_mahimahi_ptr;
    double virtual_last_mahimahi_time;
    double buffer_w;
private:
    unordered_map<int, vector<double>> vmaf_size;
    unordered_map<int, vector<int>> chunk_size;
	// void readVmaf(unordered_map<int, vector<double>> &vmaf_size);
    void readChunk(unordered_map<int, vector<int>> &chunk_size);
    std::vector<std::string> split(const std::string &s, char delim);
    void split(const std::string &s, char delim, std::vector<std::string>& result);
    int getdir(string dir, vector<string> &files);
	// void init(vector<vector<double>> &all_cooked_time, vector<vector<double>> &all_cooked_bw);
    // void load_trace(string cooked_trace_folder, vector<vector<double>> &all_cooked_time, vector<vector<double>> &all_cooked_bw, vector<string> &all_file_names);
};
