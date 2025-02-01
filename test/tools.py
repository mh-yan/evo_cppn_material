import os


# 提取文件夹地址中 output 后面的数字
def extract_output_number(folder_path):
    # 找到 output 后面的数字
    start_index = folder_path.find("output") + len("output")
    end_index = start_index
    while end_index < len(folder_path) and folder_path[end_index].isdigit():
        end_index += 1

    print(int(folder_path[start_index:end_index]))
    return int(folder_path[start_index:end_index])


# 示例文件夹地址列表
folder_paths = [
    "all_data/sym2/sym2_task_Max:E-Min:nu_0/output7",
    "all_data/sym2/sym2_task_Max:E-Min:nu_0/output10323",
    "all_data/sym2/sym2_task_Max:E-Min:nu_0/output2",
    "all_data/sym2/sym2_task_Max:E-Min:nu_0/output1",
]

# 按照 output 后面的数字排序文件夹地址
sorted_folders = sorted(folder_paths, key=extract_output_number)

# 打印排序后的结果
print(sorted_folders)
