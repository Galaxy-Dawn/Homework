import os
import pandas as pd

class CFG:
    csv_path = '/data/share/storage/annote_result'

def read_concat_csv(path,save_flag=False):
    concat_dataframe_list = []
    for sub_path in os.listdir(path):
        if sub_path.endswith('.csv'):
            sub_csv_path = os.path.join(path, sub_path)
            sub_csv = pd.read_csv(sub_csv_path)
            concat_dataframe_list.append(sub_csv)
    concat_dataframe = pd.concat(concat_dataframe_list, ignore_index=True)
    if save_flag:
        concat_dataframe.to_csv(path + 'all_batch_annote_gather.csv', index=False)
    return concat_dataframe

def describe_file(df,columns):
    num_col_stats = {}
    for col in columns:
        value_counts = df[col].value_counts()
        num_col_stats[col] = value_counts
    for col, counts in num_col_stats.items():
        print(f"Column {col} has the following numeric values:")
        for value, count in counts.items():
            if count > 20:
                print(f"{value}: {count}")

sample = read_concat_csv(CFG.csv_path)
print("Columns:",sample.columns)
print("Shape:",sample.shape)
print('Example:',sample.iloc[2])
describe_file(sample,['start time','end time'])


