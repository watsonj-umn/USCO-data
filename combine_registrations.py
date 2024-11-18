import glob
import pandas as pd

necessary_columns = ['reg_num', 'reg_date', 'creation_date', 'publication_stat', 'pub_date', 'work_type']

file_list = glob.glob('raw-data/reg_*.dta')

def process_file(file_path):

    df = pd.read_stata(file_path)
    df['reg_num'] = df['reg_num'].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=['reg_num'])
    id_cols = [col for col in necessary_columns if col in df.columns]

    claimant_cols = [col for col in df.columns if 'claimant_' in col and 'name' in col]
    corp_ind_cols = [col for col in df.columns if 'claimant_' in col and 'corp_ind' in col]

    names_df = pd.melt(df, id_vars=id_cols, value_vars=claimant_cols, var_name='claimant_info', value_name='claimant_name')
    corp_df = pd.melt(df, id_vars=id_cols, value_vars=corp_ind_cols, var_name='claimant_info', value_name='claimant_corp_ind')

    names_df['claimant_id'] = names_df['claimant_info'].str.extract('(\d+)')[0]
    corp_df['claimant_id'] = corp_df['claimant_info'].str.extract('(\d+)')[0]

    names_df.drop('claimant_info', axis=1, inplace=True)
    corp_df.drop('claimant_info', axis=1, inplace=True)

    result_df = pd.merge(names_df, corp_df, on= id_cols + ["claimant_id"])

    result_df = result_df.dropna(subset=['claimant_name'])
    result_df = result_df[result_df['claimant_corp_ind'] == "corp"]
    result_df.dropna(inplace = True)

    result_df.reset_index(drop=True, inplace=True)

    return(result_df)

frames = []

for f in file_list:
    print(f)
    df = process_file(f)
    print(df)
    frames.append(df)


combined_df = pd.concat(frames, ignore_index=True)

combined_df['claimant_name'] = combined_df['claimant_name'].str.strip().str.replace(r'\s+', ' ', regex=True)

combined_df.to_csv("combined_regs.csv")


name_counts = combined_df['claimant_name'].value_counts()
name_counts_df = name_counts.reset_index()
name_counts_df.columns = ['claimant_name', 'count']

name_counts_df.to_csv('claimant_name_counts.csv', index=False)

print("CSV files combining USCO registrations have been saved.")
