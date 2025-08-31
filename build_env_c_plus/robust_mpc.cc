#include "config.h"
#include "robust_mpc.h"

#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <algorithm>
#define MILLISECONDS_IN_SECOND  1000.0
#define B_IN_MB  1000000.0
#define BITS_IN_BYTE  8.0
#define RANDOM_SEED  42
#define VIDEO_CHUNCK_LEN  4000.0  // millisec, every time add this amount to buffer
// #define BITRATE_LEVELS  6
// #define MPC_FUTURE_CHUNK_COUNT 5 // default 8

#define A_DIM BITRATE_LEVELS
#define BUFFER_THRESH  60.0 * MILLISECONDS_IN_SECOND  // millisec, max buffer limit
#define DRAIN_BUFFER_SLEEP_TIME  500.0  // millisec
#define PACKET_PAYLOAD_PORTION  0.95
#define LINK_RTT  80  // millisec
#define PACKET_SIZE  1500  // bytes

#undef max
#undef min

//fixed video size -> envivo
// #define VIDEO_SIZE_FILE  "./envivio/video_size_"
// #define VMAF  "./envivo/vmaf/video"

#define CHUNK_TIL_VIDEO_END_CAP 48.0
#define TOTAL_VIDEO_CHUNKS 48

#define M_IN_K 1000.
// #define BITRATE_LEVELS 6
// #define REBUF_PENALTY 4.3
// #define REBUF_PENALTY 40
#define SMOOTH_PENALTY 1.0

// double VIDEO_BIT_RATE[] = {300,750,1200,1850,2850,4300};
// double VIDEO_BIT_RATE[] = {1000,2500,5000,8000,16000,40000};


void Environment::split(const std::string &s, char delim, std::vector<std::string>& result)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
	{
		result.push_back(item);
	}
}

std::vector<std::string> Environment::split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	this->split(s, delim, elems);
	return elems;
}

/*function... might want it in some class?*/
int Environment::getdir(string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		if (dirp->d_name[0] == '.') continue;  // read . or ..
		files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

Environment::Environment(std::vector<std::vector<double>> all_cooked_time, 
	std::vector<std::vector<double>> all_cooked_bw, int seed, double buffer_w, bool is_train, int mpc_future_n)
{
	this->all_cooked_time = all_cooked_time;
	this->all_cooked_bw = all_cooked_bw;
	this->is_train = is_train;

	this->mpc_future_chunk_count = mpc_future_n;

	this->video_chunk_counter = 0;
	this->buffer_size = 0;

	// pick a random trace file
	if (this->is_train)
	{
		this->trace_idx = rand() % all_cooked_time.size();
		this->cooked_time = this->all_cooked_time[this->trace_idx];
		this->cooked_bw = this->all_cooked_bw[this->trace_idx];
		// mahimahi_ptr -> cooked_bw ->  trace_idx 
		this->mahimahi_ptr =  rand() % (this->cooked_bw.size() - 1) + 1;
	}
	else
	{
		this->trace_idx = 0;
		this->cooked_time = this->all_cooked_time[this->trace_idx];
		this->cooked_bw = this->all_cooked_bw[this->trace_idx];
		this->mahimahi_ptr = 1;
	}


	// this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
	this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
	readChunk(this->chunk_size);
	// readVmaf(this->vmaf_size);
	this->virtual_mahimahi_ptr = this->mahimahi_ptr;
	this->virtual_last_mahimahi_time = this->last_mahimahi_time;
	//vector<vector<int>> CHUNK_COMBO_OPTIONS;
	this->buffer_w = buffer_w;


    for (auto idx = 0; idx < std::pow(A_DIM, this->mpc_future_chunk_count); idx++)
    {
        vector<int> vec;
        int j = idx;
        for (auto i = 0; i < this->mpc_future_chunk_count; ++i)
        {
            auto tmp = j % A_DIM;
            vec.push_back(tmp);
            j /= A_DIM;
        }
        this->CHUNK_COMBO_OPTIONS.push_back(vec);
    }
}




void Environment::reset_download_time()
{
	this->virtual_mahimahi_ptr = this->mahimahi_ptr;
	this->virtual_last_mahimahi_time = this->last_mahimahi_time;
}

double Environment::get_download_time(int video_chunk_size, double predict_tput)
{
	
	double delay = video_chunk_size / (predict_tput * B_IN_MB);
	delay += LINK_RTT / 1000.0;  // 加上 RTT（转换为秒）
	return delay;

}


int Environment::get_optimal(int last_bit_rate, double predict_tput, int horizon)
{
	//auto last_video_vmaf = this->video_chunk_vmaf0;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNKS - this->video_chunk_counter;
	auto last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1);
	auto future_chunk_length = this->mpc_future_chunk_count;
	if (TOTAL_VIDEO_CHUNKS - last_index - 1 < this->mpc_future_chunk_count)
		future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1;

	auto max_reward = -100000000;
	int send_data = 0;
	//best_combo = ()
	auto start_buffer = this->buffer_size / MILLISECONDS_IN_SECOND;
	for (auto &combo : this->CHUNK_COMBO_OPTIONS)
	{

		double curr_buffer = start_buffer;
		this->reset_download_time();


		double curr_rebuffer_time = 0.0;

		double reward_ = 0.0;

        int curr_last_bit_rate = last_bit_rate;

		for (auto position = 0; position < future_chunk_length; position++)
		{
			auto chunk_quality = combo[position];
			auto index = last_index + position + 1;
			auto sizes = this->chunk_size[chunk_quality][index];
			auto download_time = this->get_download_time(sizes, predict_tput);
			//double curr_buffer = 0.0;
			if (curr_buffer < download_time)
			{
				curr_rebuffer_time += (download_time - curr_buffer);
				curr_buffer = 0.0;
			}
			else
			{
				curr_buffer -= download_time;
			}
			curr_buffer += 4.0;
			auto reward = VIDEO_BIT_RATE[chunk_quality] / M_IN_K \
            - REBUF_PENALTY * curr_rebuffer_time \
            - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[curr_last_bit_rate]) / M_IN_K \
			- this->buffer_w * curr_buffer;

            // 更新局部变量，不影响下一次组合的初始状态
            curr_last_bit_rate = chunk_quality;

			reward_ += reward;

		}
		if (reward_ >= max_reward)
		{
			max_reward = reward_;
			send_data = combo[0];
		}
	}
	this->optimal = send_data;

	return send_data;
}

std::tuple<double, double, double, double, double, std::vector<int>, bool, int> 
Environment::get_video_chunk(int quality, bool switch_trace)
{
	auto video_chunk_size = this->chunk_size[quality][this->video_chunk_counter];
	// auto video_chunk_vmaf = this->vmaf_size[quality][this->video_chunk_counter];

	// use the delivery opportunity in mahimahi
	auto delay = 0.0;  // in ms
	auto video_chunk_counter_sent = 0;  // in bytes

	while (true)  // download video chunk over mahimahi
	{
		auto throughput = this->cooked_bw[this->mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE;
		auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;

		auto packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION;

		if (video_chunk_counter_sent + packet_payload > video_chunk_size)
		{
			auto fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION;
			delay += fractional_time;
			this->last_mahimahi_time += fractional_time;
			break;
		}
		video_chunk_counter_sent += packet_payload;
		delay += duration;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
		this->mahimahi_ptr += 1;

		if (this->mahimahi_ptr >= this->cooked_bw.size())
		{
			// loop back in the beginning
			// note: trace file starts with time 0
			this->mahimahi_ptr = 1;
			this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
		}
	}
	delay *= MILLISECONDS_IN_SECOND;
	delay += LINK_RTT;

	// rebuffer time
	auto rebuf = std::max(delay - this->buffer_size, 0.0);

	// update the buffer
	this->buffer_size = std::max(this->buffer_size - delay, 0.0);

	// add in the new chunk
	this->buffer_size += VIDEO_CHUNCK_LEN;

	// sleep if buffer gets too large
	auto sleep_time = 0.0;
	if (this->buffer_size > BUFFER_THRESH)
	{
		// exceed the buffer limit
		// we need to skip some network bandwidth here
		// but do not add up the delay
		auto drain_buffer_time = this->buffer_size - BUFFER_THRESH;
		sleep_time = std::ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME;
		this->buffer_size -= sleep_time;
		while (true)
		{
			auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;
			if (duration > sleep_time / MILLISECONDS_IN_SECOND)
			{
				this->last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND;
				break;
			}
			sleep_time -= duration * MILLISECONDS_IN_SECOND;
			this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
			this->mahimahi_ptr += 1;

			if (this->mahimahi_ptr >= this->cooked_bw.size())
			{
				// loop back in the beginning
				// note: trace file starts with time 0
				this->mahimahi_ptr = 1;
				this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
			}
		}
	}
	// the "last buffer size" return to the controller
	// Note: in old version of dash the lowest buffer is 0.
	// In the new version the buffer always have at least
	// one chunk of video
	auto return_buffer_size = this->buffer_size;

	this->video_chunk_counter += 1;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNKS - this->video_chunk_counter;

	auto end_of_video = false;
	if (this->video_chunk_counter >= TOTAL_VIDEO_CHUNKS)
	{
		end_of_video = true;
		this->buffer_size = 0;
		this->video_chunk_counter = 0;

		// this->trace_idx += 1;
		// if(this->trace_idx >= this->all_cooked_time.size())
		//    this->trace_idx = 0;
		if (this -> is_train)
		{
			this->trace_idx = rand() % this->all_cooked_time.size();
			this->cooked_time = this->all_cooked_time[this->trace_idx];
			this->cooked_bw = this->all_cooked_bw[this->trace_idx];
			// mahimahi_ptr -> cooked_bw ->  trace_idx 
			this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
		}
		else
		{
			this->trace_idx = (this->trace_idx + 1) % this->all_cooked_time.size();
			this->cooked_time = this->all_cooked_time[this->trace_idx];
			this->cooked_bw = this->all_cooked_bw[this->trace_idx];
			this->mahimahi_ptr = 1;
		}
		// this->trace_idx = rand() % all_cooked_time.size();

		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
	}

	std::vector<int> next_video_chunk_sizes;
	int bitrate_levels = this->chunk_size.size();
	for (auto i = 0; i< bitrate_levels; i++)
	{
		next_video_chunk_sizes.push_back(this->chunk_size[i][this->video_chunk_counter]);
	}


	return std::tuple<double, double, double, double, double, std::vector<int>, bool, int>
		(delay, sleep_time, return_buffer_size / MILLISECONDS_IN_SECOND, rebuf / MILLISECONDS_IN_SECOND,
		video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain);
}



void Environment::readChunk(unordered_map<int, vector<int>> &chunk_size)
{
	for (auto bitrate = 0; bitrate < BITRATE_LEVELS; bitrate++)
	{
		vector<int> tmp;
		chunk_size[bitrate] = tmp;
		ifstream fin(VIDEO_SIZE_FILE + to_string(bitrate));
		string s;
		while (getline(fin, s))
		{
			chunk_size[bitrate].push_back(stoi(s));
		}
		fin.close();
	}
}

PYBIND11_MODULE(robustmpc, m) {
    pybind11::class_<Environment>(m, "Environment")
        .def(pybind11::init<std::vector<std::vector<double>>, std::vector<std::vector<double>>, int, double, bool, int>())
		.def_readwrite("mahimahi_ptr", &Environment::mahimahi_ptr)
        .def_readwrite("trace_idx", &Environment::trace_idx)
        .def("get_optimal", &Environment::get_optimal)
        .def("get_video_chunk", &Environment::get_video_chunk);
}
