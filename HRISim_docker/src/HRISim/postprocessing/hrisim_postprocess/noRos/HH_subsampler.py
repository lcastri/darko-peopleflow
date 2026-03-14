import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import copy
from enum import Enum

class WP(Enum):
    ROOM1 = "r1"
    ROOM2 = "r2"
    CORRIDOR1 = "c1"
    CORRIDOR2 = "c2"
    CORRIDOR3 = "c3"
    CORRIDOR4 = "c4"
    CORRIDOR5 = "c5"
    CORRIDOR6 = "c6"
    CORRIDOR7 = "c7"
    CORRIDOR8 = "c8"
    CORRIDOR9 = "c9"
    CORRIDOR10 = "c10"
    CORRIDOR11 = "c11"
    CORRIDOR12 = "c12"
    CORRIDOR13 = "c13"
    CORRIDOR14 = "c14"
    CORRIDOR15 = "c15"
    CORRIDOR16 = "c16"
    CORRIDOR17 = "c17"
    CORRIDOR18 = "c18"
    CORRIDOR19 = "c19"
    CORRIDOR20 = "c20"
    CORRIDOR21 = "c21"
    CORRIDOR22 = "c22"
    CORRIDOR23 = "c23"
    CORRIDOR24 = "c24"
    CORRIDOR25 = "c25"
    CORRIDOR26 = "c26"

WPS = {wp.value: i for i, wp in enumerate(WP)}
ID_TO_WP = {v: k for k, v in WPS.items()}  
  
def plot_bandwidth(signal, varname, sampling_rate, bandwidth = None):
    fft_signal = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_signal)
    
    # Find the frequency axis
    freq_axis = np.fft.rfftfreq(len(signal), 1 / sampling_rate)
    
    # Plotting the FFT magnitude
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, fft_magnitude, label='FFT Magnitude Spectrum')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.axvline(x=bandwidth, color='red', linestyle='--')
    plt.title(f'{varname} - FFT Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show()
    
    
def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def low_pass_filter(data, cutoff_freq, sampling_rate, order=4):    
    # Validate cutoff frequency
    nyquist = 0.5 * sampling_rate
    if cutoff_freq >= nyquist:
        raise ValueError(f"Cutoff frequency ({cutoff_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz).")
    
    # Normalize cutoff frequency
    normal_cutoff = cutoff_freq / nyquist
    
    # Check data length
    if len(data) < 3 * order:
        raise ValueError("Data length must be at least 3 times the filter order.")
    
    # Design Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def get_bandwidth(df, sampling_rate, cutoff, energy_percentage, plot):
    N, dim = df.shape
    bandwidths = []
    
    for i in range(dim):
        varname = df.columns[i]
        
        # 1. Remove noise
        # filtered_signal = low_pass_filter(df.values[:,i], cutoff, sampling_rate)
        filtered_signal = moving_average_filter(df.values[:,i], 21)
        # filtered_signal = df.values[:,i]
        filtered_signal = filtered_signal[~np.isnan(filtered_signal)]
        
        # 2. Compute the FFT
        fft_values = fft(filtered_signal)
        frequencies = fftfreq(N, 1 / sampling_rate)

        # 3. Compute the power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_values[:N // 2])**2
        positive_frequencies = frequencies[:N // 2]

        # 4. Calculate total energy (sum of power spectrum)
        total_energy = np.sum(power_spectrum)

        # 5. Set target cumulative energy (e.g., 95% of total energy)
        target_energy = energy_percentage * total_energy

        # 6. Cumulatively sum the power spectrum
        cumulative_energy = np.cumsum(power_spectrum)

        # 7. Find the indices where the cumulative energy meets the target percentage
        lower_index = np.where(cumulative_energy >= (1 - energy_percentage) * total_energy)[0][0]
        upper_index = np.where(cumulative_energy >= target_energy)[0][0]

        # 8. Calculate bandwidth between first and last significant frequencies
        bandwidth = positive_frequencies[upper_index] - positive_frequencies[lower_index]

        print(f"{varname} bandwidth ({energy_percentage*100}% energy): {bandwidth} Hz")
        
        bandwidths.append(bandwidth)
        
        if plot: plot_bandwidth(filtered_signal, varname, sampling_rate, bandwidth)
    
    return max(bandwidths)


def get_initrow(df):
    for r in range(len(df)):
        if (df.iloc[r]["G_X"] != -1000 and df.iloc[r]["G_Y"] != -1000 and df.iloc[r]["R_B"] != 0 and df.iloc[r].notnull().all()):
            return r
        
        
def get_subsampling_step(cutoff = 0.5, energy_percentage=0.95, plot = True):
    wp = WP.WA_3_C.value
    rs = {}
    dfs = []
    for bag in BAGNAME:
        for tod in TOD:
            df = pd.read_csv(os.path.join(INDIR, f"{bag}", f"{bag}_{tod.value}.csv"), index_col=0)
            r = get_initrow(df)
            df = df.loc[r:, ["R_X", "R_Y", "R_V", "R_B", f"{wp}_PD",]]
            rs[tod] = r
            dfs.append(df.reset_index(drop=True))
        
    df = pd.concat(dfs, axis=0)
    df.dropna(inplace=True)

    _bw = get_bandwidth(df, SF, cutoff = cutoff, energy_percentage = energy_percentage, plot=plot)
    _ssf = 2*_bw
    _st = 1/_ssf
    _step = int(math.floor(_st * SF))
    
    del df
    return rs, _bw, _ssf, _st, _step


def get_local_pd(row):
    wp_id = int(row['R_WP'])
        
    wp_prefix = ID_TO_WP.get(wp_id)
        
    if wp_prefix:
        col_name = f"{wp_prefix}_PD"
            
        if col_name in row.index:
            return row[col_name]
                
    return 0.0


SF = 10 #Hz
INDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/csv/'
OUTDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/csv_pp/'
BAGNAME= ['DISCOVERY_BUFFET', 'DISCOVERY_POSTER', 'DISCOVERY_OFF']
# R, BW, SSF, ST, STEP = get_subsampling_step(cutoff = 0.25, energy_percentage=0.95, plot = False)
# print(f"Bandwidth fm: {BW:.4f} Hz")
# print(f"Subsampling frequency fs >= 2fm = {2*BW:.4f} Hz")
# print(f"Subsampling time 1 sample every each {ST:.4f} s")
# print(f"Subsampling step {STEP}")
print("")
STEP = 10
        
for bag in BAGNAME:
    print(f"Subsampling {bag}")
    DF = pd.read_csv(os.path.join(INDIR, f"{bag}", f"{bag}.csv"))
    
    # Remove initial rows with missing or invalid data
    r = get_initrow(DF)
    DF = pd.DataFrame(DF.values[r:,:], columns=DF.columns)
    DF.reset_index(drop=True, inplace=True)
    
    # Subsample the DataFrame using the Static method
    df_sub = DF.iloc[::STEP].copy()
    df_sub.reset_index(drop=True, inplace=True)
    
    df_sub['BC'] = -df_sub['R_B'].diff()    
    df_sub['BC'] = df_sub['BC'].fillna(0.0)
    
    # Applica la funzione al dataframe subsamplato
    df_sub['PD_WP'] = df_sub.apply(get_local_pd, axis=1)
    
    # Save specific columns      
    general_columns_name = ['TOD', 'R_WP', 'R_V', 'R_B', 'BC', 'OBS', 'PD_WP']
   
    output_dir = os.path.join(OUTDIR, f"{bag}")
    os.makedirs(output_dir, exist_ok=True)
    df_sub[general_columns_name].to_csv(os.path.join(output_dir, f"{bag}.csv"), index=False)
    print(f"Saved processed data to {output_dir}")
