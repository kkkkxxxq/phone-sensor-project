"""
HAR Pipeline - 特征工程
从窗口时序数据中提取统计特征
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft


def extract_time_domain_features(signal, prefix=''):
    """提取时域特征"""
    features = {}

    # 基本统计量
    features[f'{prefix}mean'] = np.mean(signal)
    features[f'{prefix}std'] = np.std(signal)
    features[f'{prefix}var'] = np.var(signal)
    features[f'{prefix}min'] = np.min(signal)
    features[f'{prefix}max'] = np.max(signal)
    features[f'{prefix}range'] = np.max(signal) - np.min(signal)
    features[f'{prefix}median'] = np.median(signal)
    features[f'{prefix}mad'] = np.mean(np.abs(signal - np.mean(signal)))  # 平均绝对偏差
    features[f'{prefix}iqr'] = stats.iqr(signal)  # 四分位距

    # 百分位数
    for p in [10, 25, 75, 90]:
        features[f'{prefix}p{p}'] = np.percentile(signal, p)

    # 能量
    features[f'{prefix}energy'] = np.sum(signal ** 2) / len(signal)

    # 过零率
    features[f'{prefix}zcr'] = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

    # 偏度和峰度
    if len(signal) > 2 and np.std(signal) > 1e-10:
        features[f'{prefix}skewness'] = stats.skew(signal)
        features[f'{prefix}kurtosis'] = stats.kurtosis(signal)
    else:
        features[f'{prefix}skewness'] = 0.0
        features[f'{prefix}kurtosis'] = 0.0

    return features


def extract_freq_domain_features(signal, fs=50.0, prefix=''):
    """提取频域特征"""
    features = {}
    n = len(signal)

    # FFT
    fft_vals = np.abs(fft(signal))[:n // 2]
    freqs = np.fft.fftfreq(n, 1.0 / fs)[:n // 2]

    # 频谱能量
    features[f'{prefix}freq_energy'] = np.sum(fft_vals ** 2) / len(fft_vals)

    # 频谱熵
    fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
    features[f'{prefix}freq_entropy'] = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))

    # 频谱质心
    features[f'{prefix}freq_centroid'] = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

    # 主频
    if len(fft_vals) > 0:
        dominant_idx = np.argmax(fft_vals)
        features[f'{prefix}freq_dominant'] = freqs[dominant_idx]
        features[f'{prefix}freq_dominant_mag'] = fft_vals[dominant_idx]
    else:
        features[f'{prefix}freq_dominant'] = 0.0
        features[f'{prefix}freq_dominant_mag'] = 0.0

    # 频带能量比
    total_energy = np.sum(fft_vals ** 2) + 1e-10
    bands = [(0, 3), (3, 8), (8, 15), (15, 25)]
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy = np.sum(fft_vals[mask] ** 2)
        features[f'{prefix}freq_band_{low}_{high}'] = band_energy / total_energy

    return features


def extract_window_features(window_data, feature_cols, fs=50.0):
    """从单个窗口提取所有特征"""
    features = {}

    # 对每个传感器轴提取特征
    body_accel_cols = ['ax_body', 'ay_body', 'az_body']
    grav_cols = ['ax_grav', 'ay_grav', 'az_grav']
    gyro_cols = ['gx', 'gy', 'gz']
    total_cols = ['accel_body_total', 'gyro_total']

    # 体动加速度 - 时域+频域
    for col in body_accel_cols:
        if col in feature_cols:
            idx = feature_cols.index(col)
            sig = window_data[:, idx]
            prefix = f'body_accel_{col[0]}_'
            features.update(extract_time_domain_features(sig, prefix))
            features.update(extract_freq_domain_features(sig, fs, prefix))

    # 重力加速度 - 仅时域
    for col in grav_cols:
        if col in feature_cols:
            idx = feature_cols.index(col)
            sig = window_data[:, idx]
            prefix = f'grav_{col[0]}_'
            features.update(extract_time_domain_features(sig, prefix))

    # 陀螺仪 - 时域+频域
    for col in gyro_cols:
        if col in feature_cols:
            idx = feature_cols.index(col)
            sig = window_data[:, idx]
            prefix = f'gyro_{col[0]}_'
            features.update(extract_time_domain_features(sig, prefix))
            features.update(extract_freq_domain_features(sig, fs, prefix))

    # 合成信号
    for col in total_cols:
        if col in feature_cols:
            idx = feature_cols.index(col)
            sig = window_data[:, idx]
            prefix = f'{col}_'
            features.update(extract_time_domain_features(sig, prefix))
            features.update(extract_freq_domain_features(sig, fs, prefix))

    # 加速度向量角度特征
    if all(c in feature_cols for c in ['ax_body', 'ay_body', 'az_body']):
        ax_idx = feature_cols.index('ax_body')
        ay_idx = feature_cols.index('ay_body')
        az_idx = feature_cols.index('az_body')

        ax = window_data[:, ax_idx]
        ay = window_data[:, ay_idx]
        az = window_data[:, az_idx]

        # 与各轴的夹角
        total = np.sqrt(ax**2 + ay**2 + az**2) + 1e-10
        angle_x = np.arccos(np.clip(ax / total, -1, 1))
        angle_y = np.arccos(np.clip(ay / total, -1, 1))
        angle_z = np.arccos(np.clip(az / total, -1, 1))

        features['angle_x_mean'] = np.mean(angle_x)
        features['angle_y_mean'] = np.mean(angle_y)
        features['angle_z_mean'] = np.mean(angle_z)
        features['angle_x_std'] = np.std(angle_x)
        features['angle_y_std'] = np.std(angle_y)
        features['angle_z_std'] = np.std(angle_z)

    # 陀螺仪相关性
    if all(c in feature_cols for c in ['gx', 'gy', 'gz']):
        gx = window_data[:, feature_cols.index('gx')]
        gy = window_data[:, feature_cols.index('gy')]
        gz = window_data[:, feature_cols.index('gz')]
        features['gyro_corr_xy'] = np.corrcoef(gx, gy)[0, 1] if np.std(gx) > 0 and np.std(gy) > 0 else 0
        features['gyro_corr_xz'] = np.corrcoef(gx, gz)[0, 1] if np.std(gx) > 0 and np.std(gz) > 0 else 0
        features['gyro_corr_yz'] = np.corrcoef(gy, gz)[0, 1] if np.std(gy) > 0 and np.std(gz) > 0 else 0

    return features


def extract_all_features(window_array, meta_df, feature_cols, fs=50.0):
    """提取所有窗口的特征"""
    n_windows = window_array.shape[0]
    print(f"提取特征: {n_windows} 个窗口...")

    all_features = []
    for i in range(n_windows):
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{n_windows} 窗口")
        feats = extract_window_features(window_array[i], feature_cols, fs)
        all_features.append(feats)

    print(f"  完成: {n_windows} 个窗口, 每个窗口 {len(all_features[0])} 个特征")

    # 转为DataFrame
    feat_df = pd.DataFrame(all_features)

    # 合并元数据
    result_df = pd.concat([meta_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    return result_df


def select_feature_columns(df):
    """选择特征列（排除元数据列）"""
    meta_cols = ['window_id', 'subject_id', 'activity', 'activity_cn', 'folder', 'win_idx']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return feature_cols


if __name__ == '__main__':
    import sys

    # 查找数据
    data_root = os.path.dirname(os.path.abspath(__file__))
    windows_dir = os.path.join(data_root, '..', 'data_自采', 'windows')
    windows_dir = os.path.normpath(windows_dir)

    if not os.path.exists(os.path.join(windows_dir, 'windows_data.npy')):
        # 尝试从根目录找
        windows_dir = os.path.join(data_root, 'data_自采', 'windows')

    print(f"加载窗口数据从: {windows_dir}")

    # 加载数据
    meta_df = pd.read_csv(os.path.join(windows_dir, 'windows_meta.csv'))
    window_array = np.load(os.path.join(windows_dir, 'windows_data.npy'))
    with open(os.path.join(windows_dir, 'window_features_cols.json'), 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)

    print(f"窗口数: {len(meta_df)}, 时序特征: {len(feature_cols)}")

    # 提取特征
    result_df = extract_all_features(window_array, meta_df, feature_cols, fs=50.0)

    # 保存
    output_path = os.path.join(windows_dir, 'data_features.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n特征数据已保存: {output_path}")
    print(f"形状: {result_df.shape}")

    # 特征统计
    feat_cols = select_feature_columns(result_df)
    print(f"\n特征数: {len(feat_cols)}")
    print(f"前10个特征: {feat_cols[:10]}")
