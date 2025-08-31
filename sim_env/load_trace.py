import os


# COOKED_TRACE_FOLDER = './train/'

def load_trace(cooked_trace_folders):
    if isinstance(cooked_trace_folders, str):
        cooked_trace_folders = [cooked_trace_folders]  # 确保是列表

    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []

    for folder in cooked_trace_folders:
        cooked_files = os.listdir(folder)
        for cooked_file in cooked_files:
            file_path = os.path.join(folder, cooked_file)
            if os.path.isdir(file_path):
                continue

            cooked_time = []
            cooked_bw = []

            line_count = 0
            start_time = 0
            with open(file_path, 'rb') as f:
                for line in f:
                    parse = line.split()
                    if line_count == 0:
                        start_time = float(parse[0])

                    cooked_time.append(float(parse[0]) - start_time)
                    cooked_bw.append(float(parse[1]))

                    line_count += 1
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names



# def load_trace(cooked_trace_folder):
#     cooked_files = os.listdir(cooked_trace_folder)
#     all_cooked_time = []
#     all_cooked_bw = []
#     all_file_names = []
#     for cooked_file in cooked_files:
#         # file_path = cooked_trace_folder + cooked_file
#         file_path = os.path.join(cooked_trace_folder, cooked_file)
#         if os.path.isdir(file_path):
#             continue
        
#         cooked_time = []
#         cooked_bw = []
        
#         line_count = 0
#         start_time = 0
#         # print file_path
#         with open(file_path, 'rb') as f:
#             for line in f:
#                 parse = line.split()
#                 if line_count == 0:
#                     start_time = float(parse[0])
                
#                 cooked_time.append(float(parse[0]) - start_time)
#                 cooked_bw.append(float(parse[1]))
                
#                 line_count += 1
#         all_cooked_time.append(cooked_time)
#         all_cooked_bw.append(cooked_bw)
#         all_file_names.append(cooked_file)

#     return all_cooked_time, all_cooked_bw, all_file_names
