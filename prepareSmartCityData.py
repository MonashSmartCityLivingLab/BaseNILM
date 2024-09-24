import os

import numpy as np
import pandas as pd
import scipy
import torch

appliance_mappings = {
    "athom-smart-plug-v2-f18175": "MCR",
    "athom-smart-plug-v2-f1867c": "WME",
    # "Athom-smart-plug-v2-a78696": "KET",
    "athom-smart-plug-v2-f13f8e": "LMP",
    "athom-smart-plug-v2-f16702": "TVE",
    "athom-smart-plug-v2-3ff088": "HEA",
    "athom-smart-plug-v2-3fec07": "OTH",
    "athom-smart-plug-v2-a76459": "UNK"
}

# start_datetime = '2024-07-23 21:30:00'
# end_datetime = '2024-09-08 12:40:00'
start_datetime = '2024-08-20 12:40:00'
end_datetime = '2024-08-21 12:40:00'


# Function to read and process individual CSV files
def readAndProcessCsv(file_path):
    # Read in CSV and set index
    df = pd.read_csv(file_path)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'], format='ISO8601')

    # Find the first and last non-zero consumption index
    first_non_zero_idx = df[df['Consumption(kWh)'] != 0].index[0]
    last_non_zero_idx = df[df['Consumption(kWh)'] != 0].index[-1]

    # Trim the dataframe to only include rows between the first and last non-zero consumption
    df = df.loc[first_non_zero_idx:last_non_zero_idx]

    return df


def convertKwhToW(value):
    return (value * 1000) / (1/6)


# Read in mat template
mat_template = scipy.io.loadmat('data/dataTemplate.mat')

# Append input labels
labelInp = ['time', 'id', 'P-agg']
mat_template['labelInp'] = labelInp

# Append input units
unitInp = ['sec', '-', 'W']
mat_template['unitInp'] = unitInp

# Append output labels (appliances)
labelOut = ['time', 'id']
#
# Append output units
unitOut = ['sec', '-', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
mat_template['unitOut'] = unitOut

# Read in central data csv files
# 4320 + 4320 + 1296 Records = 9936 Records
central_csv_files = [
    "data/msc-updated/central-data/Emerald_13-07-2024-11-08-2024.csv",
    "data/msc-updated/central-data/Emerald_05-08-2024-03-09-2024.csv",
    "data/msc-updated/central-data/Emerald_03-09-2024-11-09-2024.csv"
]

# Create and concatenate dataframe based on csv files
central_dataframes = [readAndProcessCsv(file) for file in central_csv_files]
combined_central_df = pd.concat(central_dataframes)

# Remove duplicates based on 'Date & Time' column, keeping the last occurrence
combined_central_df = combined_central_df.drop_duplicates(subset='Date & Time', keep='last')

# Trim dataset to provided dates
combined_central_df['Date & Time'] = pd.to_datetime(combined_central_df['Date & Time'], format='ISO8601')
combined_central_df.set_index('Date & Time', inplace=True)
combined_central_df = combined_central_df.loc[start_datetime:end_datetime]

resampled_df = combined_central_df.resample('20s').asfreq()

# Interpolate the 'Consumption(kWh)' column
resampled_df['Consumption(kWh)'] = resampled_df['Consumption(kWh)'].interpolate()

# Divide the interpolated values by 30
resampled_df['Consumption(kWh)'] /= 30

resampled_df.reset_index(inplace=True)

print(f"Start Date: {start_datetime}")
print(f"End Date: {end_datetime}")

# Create resultant input dataframe
time_counter = range(len(resampled_df))
id_col = [1] * len(resampled_df)
consumption_col = resampled_df['Consumption(kWh)'].apply(convertKwhToW)

input_df = pd.DataFrame({
    'timeCounter': time_counter,
    'id': id_col,
    'power': consumption_col
})

# Store input data in mat template
mat_template['input'] = torch.tensor(np.array(input_df))

# Read in appliance data csv
appliance_data_dirs = [x[0] for x in os.walk('data/msc-updated') if x[0].startswith('data/msc-updated/csvdata')]

# Read in appliance consumption csv's
appliance_dfs = None
for data_dir in appliance_data_dirs:
    date = data_dir.split('_')[-1]
    df = pd.read_csv(f"{data_dir}/daily_energy_consumption_payload_{date}.csv")
    if not appliance_dfs:
        appliance_dfs = {key: pd.DataFrame(columns=df.columns) for key in appliance_mappings}
    grouped_df = df.groupby('device_name')
    grouped_df_mapping = {key: value for key, value in grouped_df}

    for device_name in grouped_df_mapping:
        appliance_dfs[device_name] = pd.concat([appliance_dfs[device_name], grouped_df_mapping[device_name]], ignore_index=True)

for appliance in appliance_dfs:
    df = appliance_dfs[appliance]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values(by='timestamp')
    df.set_index('timestamp', inplace=True)

    resampled_df = df.resample('20s').agg({
        'id': 'first',
        'data': 'first',
        'device_name': 'first',
        'daily_energy_consumption': 'sum'
    })
    resampled_df = resampled_df.loc[start_datetime:end_datetime]
    resampled_df.reset_index(inplace=True)

    appliance_dfs[appliance] = resampled_df

appliance_dfs_list = appliance_dfs.values()
output_df = list(appliance_dfs_list)[0]
for i, (k, v) in enumerate(appliance_dfs.items()):
    output_df[appliance_mappings[k]] = v['daily_energy_consumption']
    labelOut.append(appliance_mappings[k])
output_df['id'] = 1
output_df['timeCounter'] = range(0, 0 + len(output_df))
output_df['OTH'] = output_df['OTH'].fillna(0)

mat_template['output'] = torch.tensor(np.array(output_df[['timeCounter', 'id', 'MCR', 'WME', 'LMP', 'TVE', 'HEA', 'OTH', 'UNK']]))
mat_template['labelOut'] = labelOut

# Save mat file
scipy.io.savemat('data/msc-updated/trainingdata.mat', mat_template)
# scipy.io.savemat('data/msc-updated/prepareddata.mat', mat_template)
