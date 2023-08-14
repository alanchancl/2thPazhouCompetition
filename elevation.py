import math
import random
import pandas as pd
import numpy as np
import pickle
import os

def merge_df(df):
    # 将计划日期和外色描述相同的连续相邻行合并起来
    result = []
    current_group = None

    for index, row in df.iterrows():
        if current_group is None:
            current_group = row
        elif (current_group[['计划日期','车型', '天窗', '外色描述', '电池特征']].equals(row[['计划日期','车型', '天窗', '外色描述', '电池特征']])):
            current_group['数量'] += row['数量']
        else:
            result.append(current_group)
            current_group = row

    # 添加最后一个分组
    if current_group is not None:
        result.append(current_group)

    merged_df = pd.DataFrame(result).reset_index(drop=True)
    return merged_df

# SA函数
def count_switches(array):
    switch_count = 0
    for arr in array:
        s = []
        for c in arr:
            parts = c.split('-')
            s.append(parts[0])

        last_char = ''
        for char in s:
            if char != last_char:
                switch_count += 1
                last_char = char
    
    return switch_count

def energy(array):
    # merged = [element for sublist in array for element in sublist]
    # KList = []
    # WList = [] 
    # CList = []
    # for s in merged:
    #     parts = s.split('-')
    #     KList.append(parts[0])
    #     WList.append(parts[1])
    #     CList.append(parts[2])


    merged_class_array = merge_adjacent_duplicates(array, 0) #聚合车型
    merged_window_array = merge_adjacent_duplicates(array, 1) #聚合电池
    merged_color_array = merge_adjacent_duplicates(array, 2) #聚合颜色
    merged_battery_array = merge_adjacent_duplicates(array, 3) #聚合电池
    k_switch_count = count_switches(merged_class_array)
    w_switch_count = count_switches(merged_window_array)
    c_switch_count = count_switches(merged_color_array)
    l_deviation, d_deviation, s_deviation, b_deviation = Calculate_Batch_Count(merged_color_array, merged_battery_array)
    d_batctout_deviation = calculate_batchout_distance_color(merged_color_array)
    fw_distance_count = calculate_batchIn_distance_FW(merged_battery_array)

    return np.array([k_switch_count, w_switch_count, c_switch_count, l_deviation, d_deviation, s_deviation, b_deviation, d_batctout_deviation,fw_distance_count])

# # SA函数
# def count_switches(s):
#     switch_count = 0
#     last_char = ''
#     for char in s:
#         if char != last_char:
#             switch_count += 1
#             last_char = char
#     return switch_count

# def energy(array):
#     merged = [element for sublist in array for element in sublist]
#     KList = []
#     WList = [] 
#     CList = []
#     for s in merged:
#         parts = s.split('-')
#         KList.append(parts[0])
#         WList.append(parts[1])
#         CList.append(parts[2])

#     k_switch_count = count_switches(KList)
#     w_switch_count = count_switches(WList)
#     c_switch_count = count_switches(CList)
#     merged_color_array = merge_adjacent_duplicates(array, 2) #聚合颜色
#     merged_battery_array = merge_adjacent_duplicates(array, 3) #聚合电池
#     l_deviation, d_deviation, s_deviation, b_deviation = Calculate_Batch_Count(merged_color_array, merged_battery_array)
#     d_batctout_deviation = calculate_batchout_distance_color(merged_color_array)
#     fw_distance_count = calculate_batchIn_distance_FW(merged_battery_array)

#     return np.array([k_switch_count, w_switch_count, c_switch_count, l_deviation, d_deviation, s_deviation, b_deviation, d_batctout_deviation,fw_distance_count])

# 评价函数计算函数
def calculate_batchout_distance_color(color_arr):
    total_deviation = 0
    for arr in color_arr:
        deviation_list = []
        for i in range(len(arr)):
            attr, count = arr[i].split('-')
            count = int(count)
            deviation = 0
            out_flag = 0
            if attr[0] == 'L' or attr[0] == 'S':
                continue
            for j in range(i+1, len(arr)):
                other_attr, other_count = arr[j].split('-')
                other_count = int(other_count)

                if (attr[0] == other_attr[0]) and (attr != other_attr):
                    out_flag = 1
                    break
                else:
                    deviation += other_count

            if out_flag==1 and deviation != 0:
                deviation_list.append(deviation)
        # print(deviation_list)
        total_deviation += sum(list(map(lambda value: max(0, 60 - value)/60, deviation_list)))
    return total_deviation

def Calculate_Batch_Count(color_arr, battery_arr):
    total_loss = 0
    l_deviation = 0
    d_deviation = 0
    s_deviation = 0
    b_deviation = 0

    for group in color_arr:
        for item in group:
            value = int(item.split('-')[1])
            if item.startswith('L'):
                l_deviation += max(0, 15 - value)/15
            elif item.startswith('D'):
                d_deviation += (0 if 0 <= value <= 4 else value - 4)/4
            elif item.startswith('S'):
                s_deviation += (0 if 15 <= value <= 30 else min(abs(value - 15), abs(value - 30)))/15
                
    for group in battery_arr:
        for item in group:
            value = int(item.split('-')[1])
            if item.startswith('G'):
                b_deviation += max(0, value - 1)

    total_loss = l_deviation + d_deviation + s_deviation + b_deviation
    # return total_loss
    return l_deviation, d_deviation, s_deviation, b_deviation

def merge_adjacent_duplicates(arr, key):
    merged_arr = []
    for _, arr in enumerate(arr):
        merged_day_arr = []
        
        i = 0
        while i < len(arr):
            current_item = arr[i]
            current_parts = current_item.split('-')
            # current_key = '-'.join(current_parts[:-1])  # 去掉数量部分
            current_key = current_parts[key]
            current_quantity = int(current_parts[-1])    # 获取数量
            
            j = i + 1
            while j < len(arr):
                next_item = arr[j]
                next_parts = next_item.split('-')
                # next_key = '-'.join(next_parts[:-1])      # 去掉数量部分
                next_key = next_parts[key]
                next_quantity = int(next_parts[-1])        # 获取数量
                
                if current_key == next_key:
                    current_quantity += next_quantity
                    j += 1
                else:
                    break
            
            merged_day_arr.append(f"{current_key}-{current_quantity}")
            i = j
        merged_arr.append(merged_day_arr)
    return merged_arr

def calculate_batchIn_distance_FW(battery_arr):
    total_deviation = 0
    for arr in battery_arr:
        deviation_list = []
        for i in range(len(arr)):
            attr, count = arr[i].split('-')
            if attr != 'FW':
                continue
            count = int(count)
            deviation = 0
            out_flag = 0

            for j in range(i+1, len(arr)):
                other_attr, other_count = arr[j].split('-')
                other_count = int(other_count)

                if attr == other_attr:
                    out_flag = 1
                    break
                else:
                    deviation += other_count

            if out_flag==1 and deviation != 0:
                deviation_list.append(deviation)
            
        total_deviation += sum(deviation_list)
    return total_deviation

def calculate_batchIn_distance_FW(battery_arr):
    total_deviation = 0
    for arr in battery_arr:
        deviation_list = []
        for i in range(len(arr)):
            attr, count = arr[i].split('-')
            if attr != 'FW':
                continue
            count = int(count)
            deviation = 0
            out_flag = 0

            for j in range(i+1, len(arr)):
                other_attr, other_count = arr[j].split('-')
                other_count = int(other_count)

                if attr == other_attr:
                    out_flag = 1
                    break
                else:
                    deviation += other_count

            if out_flag==1 and deviation != 0:
                deviation_list.append(deviation)
                
     
        total_deviation += sum(deviation_list)
    return total_deviation

df = pd.read_csv(
    r'C:\Users\alanc\Desktop\featurize-output\work\outB\2023-08-08 15-27-24\commit.csv',
    header=0)
def data_process(df):
    # 获取颜色种类
    unique_colors = df['外色描述'].unique()
    # 获取电池种类/电池类型映射
    unique_battery = df['电池特征'].unique()

    # 将"计划日期"列转换为日期类型
    df['计划日期'] = pd.to_datetime(df['计划日期'])

    # 创建标签映射字典
    label_mapping = {
        '无天窗': 'W1',
        '整体式全景天窗': 'W2',
        '天幕': 'W3',
        '全景EC天幕': 'W4'
    }

    color_mapping = {
        '量子红': 'S1', '量子红-Y': 'S1', '冰玫粉': 'S3', '冰玫粉-Y': 'S3', '蒂芙尼蓝': 'S5', '星漫绿': 'S6',
        '星漫绿-Y': 'S6', '琉璃红': 'S8', '夜荧黄': 'S9', '黄绿荧': 'S10', '薄荷贝绿': 'S11', '烟雨青': 'S12',
        '幻光紫': 'S13', '广交红': 'S14', '闪电橙': 'S15', '脉冲蓝': 'S16', '天际灰': 'S17', '火焰橙': 'S18',
        '幻光紫-Y': 'S13', '琉璃红': 'S20', '松花黄': 'S21', '松花黄-Y': 'S21',
        
        '白云蓝': 'L1', '极地白': 'L2', '极地白-Y': 'L2', '幻影银': 'L4', '幻影银(出租车)': 'L4',
        '极速银': 'L6', '极速银-Y': 'L6', '极速银(出租车)': 'L6', '夜影黑': 'L9', '夜影黑-Y': 'L19',
        '自由灰': 'L11', '自由灰-Y': 'L11', '素雅灰': 'L13', '素雅灰-Y': 'L13', '天青色': 'L15', '天青色-Y': 'L15',
        '珍珠白': 'L17', '全息银': 'L18',
        
        '黑/冰玫粉': 'D1', '黑/全息银': 'D2', '黑/极地白-Y': 'D3', '黑/星漫绿': 'D4', '黑/极地白': 'D3', '黑/天青色-Y': 'D6',
        '黑/量子红': 'D7', '黑/烟雨青': 'D8', '黑/幻光紫': 'D9',
    }

    battery_mapping = {battery: 'FW' if battery == '厂商A+厂商B 93.3kWh+180/180kW'
                    else 'GB' if battery == '厂商G(石墨烯)+厂商B 70.4kWh+165kW'
                    else 'CB' for battery in unique_battery} 

    # 使用 replace 函数进行转换
    df['天窗'] = df['天窗'].replace(label_mapping)
    df['外色描述'] = df['外色描述'].replace(color_mapping)
    df['电池特征'] = df['电池特征'].replace(battery_mapping)
    df['数量'] = 1

    new_result = merge_df(df[['计划日期','车型', '天窗', '外色描述', '电池特征','数量']])
    new_result['新组合'] = new_result['车型'] + '-' + new_result['天窗'] + '-' + new_result['外色描述'] + '-' + new_result['电池特征'] + '-' + new_result['数量'].astype(str)

    # 按计划日期分组，统计每个组合的数量
    # new_result = new_result.groupby('计划日期')['新组合'].agg([('组合种类数量', 'unique')])
    grouped = new_result.groupby('计划日期')['新组合'].apply(list)
    raw_array = [np_array for np_array in grouped]
    return raw_array

    # data_array = [np_array.tolist() for np_array in new_result['组合种类数量']]

# # 使用pickle.dump保存数组到文件
# with open('test_array.pkl', 'wb') as f:
#     pickle.dump(data_array, f)


def get_subfolder_paths(folder_path):
    subfolder_paths = []
    subfolder_names = []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            subfolder_paths.append(os.path.join(folder_path, entry.name))
            subfolder_names.append(entry.name)
    return subfolder_paths, subfolder_names

if __name__ == '__main__':
    # 指定文件夹路径
    folder_path = "fold"

    # 获取所有子文件夹的完整路径
    subfolder_paths, subfolder_names = get_subfolder_paths(folder_path)
    LOSS = []

    for _, path in enumerate(subfolder_paths):
        csv_path = os.path.join(path, 'answer.csv')
        df = pd.read_csv(csv_path, header=0)
        data_array = data_process(df)
        weight = np.array([50, 4, 2, 1, 1, 1, 1, 1, 1])
        current_energy = sum(energy(data_array)*weight)
        LOSS.append(current_energy)

    # 转换为DataFrame
    df = pd.DataFrame({'Score': subfolder_names, 'My_Loss': LOSS})
    # 按照 My_Loss 列排序
    sorted_df = df.sort_values(by='My_Loss')
    print(df)
