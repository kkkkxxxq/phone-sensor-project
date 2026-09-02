"""
HAR Pipeline - 数据导入与窗口切分
基于自采的手机传感器数据（加速度计+陀螺仪）
"""
import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

# 活动标签映射
ACTIVITY_MAP = {
    'WALKING': 'WALKING',
    'WALKING_UPSTAIRS': 'WALKING_UPSTAIRS',
    'WALKING_DOWNSTAIRS': 'WALKING_DOWNSTAIRS',
    'SITTING': 'SITTING',
    'STANDING': 'STANDING',
    'LAYING': 'LAYING',
}

ACTIVITY_CN = {
    'WALKING': '走路',
    'WALKING_UPSTAIRS': '爬楼梯',
    'WALKING_DOWNSTAIRS': '下楼梯',
    'SITTING': '静坐',
    'STANDING': '静站',
    'LAYING': '躺卧',
}


def detect_activity_from_folder(folder_name):
    """从文件夹名推断活动类型"""
    name = folder_name.lower()
    if '爬楼梯' in folder_name or 'upstairs' in name:
        return 'WALKING_UPSTAIRS'
    elif '下楼' in folder_name or 'downstairs' in name:
        return 'WALKING_DOWNSTAIRS'
    elif '走路' in folder_name or 'walking' in name:
        return 'WALKING'
    elif '静坐' in folder_name or 'sitting' in name:
        return 'SITTING'
    elif '静站' in folder_name or 'standing' in name:
        return 'STANDING'
    elif '躺卧' in folder_name or 'laying' in name:
        return 'LAYING'
    return None


def detect_subject_from_folder(folder_name):
    """从文件夹名推断受试者ID"""
    # 匹配 A1, A2, A, C 等
    match = re.search(r'([AC])(\d*)', folder_name)
    if match:
        # A1, A2 都归为 A 组
        return match.group(1)
    return 'unknown'


def find_csv_files(recording_dir):
    """查找加速度计和陀螺仪CSV文件，处理不同的目录结构"""
    accel_file = None
    gyro_file = None
    magnet_file = None

    # 直接在目录下找
    for f in os.listdir(recording_dir):
        if 'Accelerometer' in f and f.endswith('.csv'):
            accel_file = os.path.join(recording_dir, f)
        elif 'Gyroscope' in f and f.endswith('.csv'):
            gyro_file = os.path.join(recording_dir, f)
        elif 'Magnetometer' in f and f.endswith('.csv'):
            magnet_file = os.path.join(recording_dir, f)

    # 如果没找到，检查子目录（跳过__MACOSX）
    if accel_file is None or gyro_file is None:
        for root, dirs, files in os.walk(recording_dir):
            if '__MACOSX' in root:
                continue
            for f in files:
                if 'Accelerometer' in f and f.endswith('.csv') and accel_file is None:
                    accel_file = os.path.join(root, f)
                elif 'Gyroscope' in f and f.endswith('.csv') and gyro_file is None:
                    gyro_file = os.path.join(root, f)
                elif 'Magnetometer' in f and f.endswith('.csv') and magnet_file is None:
                    magnet_file = os.path.join(root, f)

    return accel_file, gyro_file, magnet_file


def load_phyphox_csv(filepath):
    """加载phyphox导出的CSV文件"""
    if filepath is None or not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        # 重命名列
        cols = df.columns.tolist()
        rename_dict = {}
        for col in cols:
            if 'Time' in col and 's' in col:
                rename_dict[col] = 'time'
            elif 'Acceleration x' in col:
                rename_dict[col] = 'ax'
            elif 'Acceleration y' in col:
                rename_dict[col] = 'ay'
            elif 'Acceleration z' in col:
                rename_dict[col] = 'az'
            elif 'Gyroscope x' in col:
                rename_dict[col] = 'gx'
            elif 'Gyroscope y' in col:
                rename_dict[col] = 'gy'
            elif 'Gyroscope z' in col:
                rename_dict[col] = 'gz'
        df = df.rename(columns=rename_dict)
        return df
    except Exception as e:
        print(f"  加载失败 {filepath}: {e}")
        return None


def merge_sensor_data(accel_df, gyro_df, target_freq=50.0):
    """合并加速度计和陀螺仪数据，按时间对齐并重采样"""
    if accel_df is None or gyro_df is None:
        return None

    # 计算共同时间范围
    t_min = max(accel_df['time'].min(), gyro_df['time'].min())
    t_max = min(accel_df['time'].max(), gyro_df['time'].max())

    if t_max <= t_min:
        return None

    # 创建统一时间轴
    duration = t_max - t_min
    n_samples = int(duration * target_freq)
    if n_samples < 100:
        return None

    new_time = np.linspace(t_min, t_max, n_samples)

    # 重采样
    merged = pd.DataFrame({'time': new_time})
    for col in ['ax', 'ay', 'az']:
        merged[col] = np.interp(new_time, accel_df['time'], accel_df[col])
    for col in ['gx', 'gy', 'gz']:
        merged[col] = np.interp(new_time, gyro_df['time'], gyro_df[col])

    return merged


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    """Butterworth滤波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype)
    return signal.filtfilt(b, a, data)


def separate_gravity(merged_df, fs=50.0):
    """分离重力分量和体动分量"""
    df = merged_df.copy()
    # 0.3Hz低通 = 重力分量
    for axis in ['ax', 'ay', 'az']:
        df[f'{axis}_grav'] = butterworth_filter(df[axis], 0.3, fs, order=4, btype='low')
        df[f'{axis}_body'] = df[axis] - df[f'{axis}_grav']
    # 计算合成加速度
    df['accel_total'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['accel_body_total'] = np.sqrt(df['ax_body']**2 + df['ay_body']**2 + df['az_body']**2)
    df['gyro_total'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    return df


def sliding_window(data, window_size, step_size):
    """滑动窗口切分"""
    windows = []
    n = len(data)
    start = 0
    while start + window_size <= n:
        windows.append(data.iloc[start:start + window_size])
        start += step_size
    return windows


def process_all_recordings(data_root, output_dir, window_size=128, step_size=64, target_freq=50.0):
    """
    处理所有采集记录
    返回: window_df (每个窗口一行特征列 + 标签 + subject_id)
    """
    print(f"数据根目录: {data_root}")
    print(f"窗口大小: {window_size} samples ({window_size/target_freq:.1f}s)")
    print(f"步长: {step_size} samples ({step_size/target_freq:.1f}s)")
    print(f"目标采样率: {target_freq} Hz")
    print("-" * 60)

    all_windows = []
    recording_info = []
    window_id_counter = 0

    # 遍历所有子文件夹
    folders = sorted([f for f in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, f))
                      and not f.startswith('.')
                      and not f.startswith('HAR_')
                      and not f == 'skill'
                      and not f == 'code'
                      and not f == 'streamlit_app'
                      and not f == 'data_自采'])

    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        activity = detect_activity_from_folder(folder)
        subject = detect_subject_from_folder(folder)

        if activity is None:
            continue

        print(f"\n处理: {folder}")
        print(f"  受试者: {subject}, 活动: {activity} ({ACTIVITY_CN.get(activity, '')})")

        accel_file, gyro_file, magnet_file = find_csv_files(folder_path)
        print(f"  加速度计: {accel_file}")
        print(f"  陀螺仪: {gyro_file}")

        accel_df = load_phyphox_csv(accel_file)
        gyro_df = load_phyphox_csv(gyro_file)

        if accel_df is None:
            print(f"  跳过: 无加速度计数据")
            continue
        if gyro_df is None:
            print(f"  跳过: 无陀螺仪数据")
            continue

        # 合并并对齐
        merged = merge_sensor_data(accel_df, gyro_df, target_freq)
        if merged is None or len(merged) < window_size:
            print(f"  跳过: 数据太短 ({len(merged) if merged is not None else 0} samples)")
            continue

        # 分离重力
        merged = separate_gravity(merged, target_freq)

        duration = merged['time'].iloc[-1] - merged['time'].iloc[0]
        actual_freq = len(merged) / duration
        print(f"  时长: {duration:.1f}s ({duration/60:.1f}min), 样本数: {len(merged)}, 采样率: {actual_freq:.1f}Hz")

        # 切窗
        windows = sliding_window(merged, window_size, step_size)
        print(f"  窗口数: {len(windows)}")

        recording_info.append({
            'folder': folder,
            'subject_id': subject,
            'activity': activity,
            'activity_cn': ACTIVITY_CN.get(activity, ''),
            'duration_s': round(duration, 1),
            'n_samples': len(merged),
            'sampling_rate': round(actual_freq, 1),
            'n_windows': len(windows),
        })

        # 保存每个窗口（先存为numpy数组用于后续特征提取）
        for i, win in enumerate(windows):
            window_id = f"win_{window_id_counter:06d}"
            window_id_counter += 1
            all_windows.append({
                'window_id': window_id,
                'subject_id': subject,
                'activity': activity,
                'activity_cn': ACTIVITY_CN.get(activity, ''),
                'folder': folder,
                'win_idx': i,
                'data': win,
            })

    print("\n" + "=" * 60)
    print(f"总记录数: {len(recording_info)}")
    print(f"总窗口数: {len(all_windows)}")

    # 保存采集信息
    info_df = pd.DataFrame(recording_info)
    info_path = os.path.join(output_dir, 'recording_info.csv')
    info_df.to_csv(info_path, index=False, encoding='utf-8-sig')
    print(f"\n采集信息已保存: {info_path}")

    return all_windows, info_df


def save_windows_data(all_windows, output_dir):
    """保存窗口数据为CSV和numpy格式"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存窗口元数据
    meta_list = []
    for w in all_windows:
        meta_list.append({
            'window_id': w['window_id'],
            'subject_id': w['subject_id'],
            'activity': w['activity'],
            'activity_cn': w['activity_cn'],
            'folder': w['folder'],
            'win_idx': w['win_idx'],
        })
    meta_df = pd.DataFrame(meta_list)
    meta_path = os.path.join(output_dir, 'windows_meta.csv')
    meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"窗口元数据已保存: {meta_path}")

    # 保存窗口时序数据（numpy格式，节省空间）
    # 形状: (n_windows, window_size, n_features)
    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz',
                    'ax_grav', 'ay_grav', 'az_grav',
                    'ax_body', 'ay_body', 'az_body',
                    'accel_total', 'accel_body_total', 'gyro_total']

    n_windows = len(all_windows)
    window_size = len(all_windows[0]['data'])
    n_features = len(feature_cols)

    window_array = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    for i, w in enumerate(all_windows):
        for j, col in enumerate(feature_cols):
            window_array[i, :, j] = w['data'][col].values

    npy_path = os.path.join(output_dir, 'windows_data.npy')
    np.save(npy_path, window_array)
    print(f"窗口时序数据已保存: {npy_path} (shape: {window_array.shape})")

    # 保存特征列名
    cols_path = os.path.join(output_dir, 'window_features_cols.json')
    with open(cols_path, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    return meta_df, window_array, feature_cols


if __name__ == '__main__':
    # 数据根目录
    data_root = os.path.dirname(os.path.abspath(__file__))
    # 向上找一级
    if not os.path.exists(os.path.join(data_root, '走路15minA1')):
        data_root = os.path.dirname(data_root)

    output_dir = os.path.join(data_root, 'data_自采', 'windows')
    os.makedirs(output_dir, exist_ok=True)

    all_windows, info_df = process_all_recordings(data_root, output_dir)
    meta_df, window_array, feature_cols = save_windows_data(all_windows, output_dir)

    print("\n按受试者统计:")
    print(meta_df.groupby('subject_id').size())
    print("\n按活动统计:")
    print(meta_df.groupby(['activity', 'activity_cn']).size())
    print("\n按受试者×活动统计:")
    print(meta_df.groupby(['subject_id', 'activity']).size().unstack(fill_value=0))
