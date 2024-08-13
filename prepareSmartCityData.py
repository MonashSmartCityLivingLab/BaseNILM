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

start_datetime ='2024-07-23 21:30:00'
end_datetime = '2024-08-11 12:40:00'

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

# Read in central data csv
central_df = pd.read_csv("data/msc/central-data/Emerald_13-07-2024-11-08-2024.csv")

# Trim starting zeros from dataframe
first_non_zero_index = 1569  # Hardcoded for sample data
central_df = central_df.iloc[first_non_zero_index:].reset_index(drop=True)

# Trim ending zeros from dataframe
last_non_zero_index = central_df[central_df['Consumption(kWh)'] != 0].index[-1]
central_df = central_df.iloc[:last_non_zero_index]

# Trim dataset to provided dates
central_df['Date & Time'] = pd.to_datetime(central_df['Date & Time'], format='ISO8601')
central_df.set_index('Date & Time', inplace=True)
central_df = central_df.loc[start_datetime:end_datetime]
central_df.reset_index(inplace=True)

# Get start and end timestamps
# start_datetime = central_df['Date & Time'].iloc[0]
# end_datetime = central_df['Date & Time'].iloc[-1]

print(f"Start Date: {start_datetime}")
print(f"End Date: {end_datetime}")

# Create resultant input dataframe
time_counter = range(len(central_df))
id_col = [1] * len(central_df)
consumption_col = central_df['Consumption(kWh)'].apply(convertKwhToW)

input_df = pd.DataFrame({
    'timeCounter': time_counter,
    'id': id_col,
    'power': consumption_col
})

# Store input data in mat template
mat_template['input'] = torch.tensor(np.array(input_df))

# Read in appliance data csv
appliance_data_dirs = [x[0] for x in os.walk('data/msc') if x[0].startswith('data/msc/csvdata')]

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

    resampled_df = df.resample('10min').agg({
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

mat_template['output'] = torch.tensor(np.array(output_df[['timeCounter', 'id', 'MCR', 'WME', 'LMP', 'TVE', 'HEA', 'OTH', 'UNK']]))
mat_template['labelOut'] = labelOut

# # Save mat file
scipy.io.savemat('data/msc/prepareddata.mat', mat_template)