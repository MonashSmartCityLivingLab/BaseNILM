import numpy as np
import pandas as pd
import scipy
import torch


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
# labelOut = ['time', 'id', 'FRE', 'AC1', 'AC2', 'WME', 'CPU', 'IRN', 'UNK', 'TVE', 'WET']
# mat_template['labelOut'] = labelOut
#
# Append output units
# unitOut = ['sec', '-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
# mat_template['unitOut'] = unitOut

# Read in central data csv
central_df = pd.read_csv("data/monash-smart-city/central-data/Emerald_13-07-2024-11-08-2024.csv")

# Trim starting zeros from dataframe
first_non_zero_index = 1569  # Hardcoded for sample data
central_df = central_df.iloc[first_non_zero_index:].reset_index(drop=True)

# Trim ending zeros from dataframe
last_non_zero_index = central_df[central_df['Consumption(kWh)'] != 0].index[-1]
central_df = central_df.iloc[:last_non_zero_index].reset_index(drop=True)

print(central_df.head(10))

# Get start and end timestamps
start_datetime = central_df['Date & Time'].iloc[0]
end_datetime = central_df['Date & Time'].iloc[-1]

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

# Save mat file
scipy.io.savemat('data/monash-smart-city/converted-data/prepared-data.mat', mat_template)

# Read in appliance data csv