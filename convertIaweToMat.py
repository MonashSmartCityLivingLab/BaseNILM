import numpy as np
import pandas as pd
import scipy
import h5py
import torch
import hdf5plugin

column_names = [
    'power_active', 'power_reactive', 'power_apparent',
    'frequency', 'voltage', 'power_factor', 'current'
]

start_date = '2013-06-07'
end_date = '2013-06-28'
date_range = pd.date_range(start=start_date, end=end_date, freq='1s')


def fill_gaps(df, date_range):
    df = df.set_index('timestamp').reindex(date_range).fillna(0).reset_index()
    return df


# Read in mat template
mat_template = scipy.io.loadmat('data/dataTemplate.mat')

# Append input labels
labelInp = ['time', 'id', 'P-agg']
mat_template['labelInp'] = labelInp

# Append input units
unitInp = ['sec', '-', 'VAr']
mat_template['unitInp'] = unitInp

# Append output labels (appliances)
labelOut = ['time', 'id', 'FRE', 'AC1', 'AC2', 'WME', 'CPU', 'IRN', 'UNK', 'TVE', 'WET']
mat_template['labelOut'] = labelOut
#
# Append output units
unitOut = ['sec', '-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
mat_template['unitOut'] = unitOut

# Append central power data to input table
with h5py.File('data/iawe/iawe.h5', 'r') as h5file:
    # Extract data from meter 1
    meter1_table = h5file['building1']['elec']['meter1']['table']
    meter1_values = meter1_table['values_block_0'][:]
    meter1_df = pd.DataFrame(meter1_values, columns=column_names)
    meter1_df['timestamp'] = pd.to_datetime(meter1_table['index'])
    meter1_df.set_index('timestamp')
    meter1_filled_df = fill_gaps(meter1_df, date_range)

    # Extract data from meter 2
    meter2_table = h5file['building1']['elec']['meter2']['table']
    meter2_values = meter2_table['values_block_0'][:]
    meter2_df = pd.DataFrame(meter2_values, columns=column_names)
    meter2_df['timestamp'] = pd.to_datetime(meter2_table['index'])
    meter2_df.set_index('timestamp')
    meter2_filled_df = fill_gaps(meter2_df, date_range)

    combined_filled_df = pd.merge_asof(meter2_filled_df, meter1_filled_df, on='index', direction='nearest', suffixes=[None, '_2']).assign(power=lambda d: d['power_apparent'].add(d.pop('power_apparent_2')))
    combined_filled_df['id'] = 1
    combined_filled_df['timeCounter'] = range(0, 0+len(combined_filled_df))
    print(combined_filled_df[15:25])

mat_template['input'] = torch.tensor(np.array(combined_filled_df[['timeCounter', 'id', 'power']]))

# Append appliance data to output table
appliance_data = []
with h5py.File('data/iawe/iawe.h5', 'r') as h5file:
    # Output keys are in order from metres 3 to 12
    appliance_table_list = []
    appliance_value_list = []
    for i in range(3, 12):
        appliance_table_list.append(h5file['building1']['elec'][f'meter{str(i)}']['table'])
        appliance_value_list.append(h5file['building1']['elec'][f'meter{str(i)}']['table']['values_block_0'])

    appliance_dataframes = []
    filled_dataframes = []
    for i in range(0, len(appliance_value_list)):
        df = pd.DataFrame(appliance_value_list[i], columns=column_names)
        df['timestamp'] = pd.to_datetime(appliance_table_list[i]['index'])
        df.set_index('timestamp')
        appliance_dataframes.append(df)

    filled_dataframes = [fill_gaps(df, date_range) for df in appliance_dataframes]

    output_df = filled_dataframes[0]
    for i in range(0, len(filled_dataframes)):
        output_df[labelOut[i+2]] = filled_dataframes[i]['power_active']
    output_df['id'] = 1
    output_df['timeCounter'] = range(0, 0+len(output_df))


mat_template['output'] = torch.tensor(np.array(output_df[['timeCounter', 'id', 'FRE', 'AC1', 'AC2', 'WME', 'CPU', 'IRN', 'UNK', 'TVE', 'WET']]))

scipy.io.savemat('data/iawe/iawe5.mat', mat_template)
