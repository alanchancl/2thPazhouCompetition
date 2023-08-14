import math
import random
import pandas as pd
import numpy as np
import pickle
import datetime
# 评价函数计算函数

def count_array_num(array):
    merged = [element for sublist in array for element in sublist]
    num = 0 
    for s in merged:
        parts = s.split('-')
        num += int(parts[-1])
    print(num)
    return num

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

#函数区
def distribute(n, max_batch_size):

    # 计算需要的车辆数
    if n % max_batch_size != 0:
        num_full_batches = (n // max_batch_size) + 1
    else:
        num_full_batches = n // max_batch_size
    
    # 计算每车装满的货物数量
    full_batch_size = n // num_full_batches
    
    # 计算剩余的货物数量
    remaining_cargo = n % num_full_batches
        
    batches = []
    
    # 将满批次货物均匀分配
    for i in range(num_full_batches):
        batches.append(full_batch_size)
    
    # 将剩余的货物均匀分配到每车
    for i in range(remaining_cargo):
        batches[i] = batches[i] + 1
    return batches

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


def swap_within_subarray(subarray):
    i, j = random.sample(range(len(subarray)), 2)
    subarray[i], subarray[j] = subarray[j], subarray[i]

def simulated_annealing(array, initial_temperature, cooling_rate, iterations):
    weight = np.array([50, 4, 2, 1, 1, 1, 1, 1, 1])
    current_energy = sum(energy(array)*weight)
    print('current_energy:', current_energy)
    subarray_count = len(array)
    
    for i in range(iterations):
        temperature = initial_temperature / (1 + cooling_rate * i)
        
        # Choose a random subarray to swap within
        subarray_index = random.randint(0, subarray_count - 1)
        new_array = [sublist[:] for sublist in array]  # Copy current array
        
        # Swap within chosen subarray
        if len(new_array[subarray_index]) == 1:
            continue

        swap_within_subarray(new_array[subarray_index])
        
        new_energy = sum(energy(new_array)*weight)
        energy_difference = new_energy - current_energy
        print('{}:    current_energy:{}'.format(i, current_energy))
        if energy_difference < 0 or random.random() < math.exp(-energy_difference / temperature):
            array = new_array
            current_energy = new_energy

    print('new_energy:', current_energy)
    return array, current_energy


def find_keys_by_value(mapping, target_value):
    keys = []
    for key, value in mapping.items():
        if value == target_value:
            keys.append(key)
    return keys


def SortxlsxAndOutputZip(df, sorted_array, mapping, path_name):
    # 创建一个新的DataFrame用于存储抽取的结果
    extracted_data = pd.DataFrame()
    date_list = df['计划日期'].unique()
    for idx, group in enumerate(sorted_array):
        date_name = date_list[idx]
        for item in group:
            Kind, Window, Color, Battery, Number = item.split('-')
            Kind_name = Kind
            Window_name = find_keys_by_value(mapping[0], Window)
            Color_name = find_keys_by_value(mapping[1], Color)
            Battery_name = find_keys_by_value(mapping[2], Battery)
            Class_Num = int(Number)
            # if Color == 'S13':
            filtered_df = df[
                (df['计划日期'] == date_name) &
                (df['车型'] == Kind_name) &
                (df['天窗'].isin(Window_name)) &
                (df['外色描述'].isin(Color_name)) &
                (df['电池特征'].isin(Battery_name))
            ]
            # filtered_df = filtered_df.sort_values(by=['外色描述','车辆等级描述','电池特征','内饰描述','序号'])

            filtered_indices = filtered_df["序号"].values
            if len(filtered_df) == Class_Num:
                # extracted_data.append(filtered_df)
                # index_list.extend(filtered_df.index.tolist())
                extracted_row = df[df["序号"].isin(filtered_indices)].sort_values(by=['外色描述','车辆等级描述','电池特征','内饰描述','序号'])
                extracted_data = pd.concat([extracted_data, extracted_row])
                df = df.drop(extracted_row.index)
            else:
                filtered_indices = filtered_indices[:Class_Num]
                extracted_row = df[df["序号"].isin(filtered_indices)].sort_values(by=['外色描述','车辆等级描述','电池特征','内饰描述','序号'])
                extracted_data = pd.concat([extracted_data, extracted_row])
                df = df.drop(extracted_row.index)

    ## 输出结果
    from io import StringIO

    buf = StringIO()
    extracted_data.to_csv(buf,index=False)

    from zipfile import ZipFile
    with ZipFile('result/{}.zip'.format(path_name), 'w') as myzip:
            with myzip.open('answer.csv', 'w') as myfile:
                myfile.write(buf.getvalue().encode())

if __name__ == '__main__':
    #读取数据
    df = pd.read_excel(
        'B榜-工厂智能排产算法赛题.xlsx',
        sheet_name='原数据',
        header=0,
        engine='openpyxl')

    df_raw = df.copy()

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

    result1 = df.groupby(['计划日期','车型', '天窗', '外色描述', '电池特征']).size().reset_index(name='数量')

    new_rows = []
    for index, row in result1.iterrows():
        if ('S' in row['外色描述']) & (row['数量'] > 30):
            batches = distribute(row['数量'], 30)
            for i in range(len(batches)):
                new_row = row.copy()
                new_row['数量'] = batches[i]
                new_rows.append(new_row)

        elif ('D' in row['外色描述']) & (row['数量'] > 4):
            batches = distribute(row['数量'], 4)
            for i in range(len(batches)):
                new_row = row.copy()
                new_row['数量'] = batches[i]
                new_rows.append(new_row)

        elif row['电池特征'] == 'GB' and row['数量'] > 1:
            batches = distribute(row['数量'], 1)
            for i in range(len(batches)):
                new_row = row.copy()
                new_row['数量'] = batches[i]
                new_rows.append(new_row)
        else:
            new_rows.append(row)

    new_result = pd.DataFrame(new_rows).reset_index(drop=True)

    # 将车型和车窗组合为一个新列
    new_result['新组合'] = new_result['车型'] + '-' + new_result['天窗'] + '-' + new_result['外色描述'] + '-' + new_result['电池特征'] + '-' + new_result['数量'].astype(str)

    # 按计划日期分组，统计每个组合的数量
    grouped = new_result.groupby('计划日期')['新组合'].apply(list)
    # new_result1 = new_result.groupby('计划日期')['新组合'].agg([('组合种类数量', 'unique')])
    raw_array = [np_array for np_array in grouped]

    # print(raw_array)
    # count_array_num(raw_array)

    initial_temperature = 100
    cooling_rate = 0.1
    iterations = 500000
    optimized_array, optimized_energy = simulated_annealing(raw_array, initial_temperature, cooling_rate, iterations)

    # 指定保存文件的路径和文件名
    current_time = datetime.datetime.now()
    file_name = '{}-{}-{}-{}-{}'.format(current_time.strftime("%Y-%m-%d_%H-%M-%S"), round(optimized_energy, 4), iterations, cooling_rate, initial_temperature)
    file_path = 'result/{}.pkl'.format(file_name)
    # 使用pickle.dump保存数组到文件
    with open(file_path, 'wb') as f:
        pickle.dump(optimized_array, f)
    
    SortxlsxAndOutputZip(df_raw, optimized_array, [label_mapping, color_mapping, battery_mapping], file_name)

