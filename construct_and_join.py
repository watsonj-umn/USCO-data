import pandas as pd
import numpy as np


def classify_industry(naics):
    if pd.isna(naics) or naics == '':
        return 'Others'
    
    naics_str = str(int(naics)) if isinstance(naics, (int, float)) else str(naics)
    it_prefixes = ["3341", "3342", "3344", "3345", "3346", "3353", "5112", "5141", "5171", "5172", "5179", "5182", "5191", "5413", "5414", "5415", "5416", "5142", "5187", "5133", "5177"]
    life_science_prefixes = ["3254", "3391", "5417"]
    retail_prefixes = ["42", "44", "45"]
    
    if naics_str.startswith('51'):
        return 'Media'
    elif any(naics_str.startswith(prefix) for prefix in it_prefixes):
        return 'IT'
    elif any(naics_str.startswith(prefix) for prefix in life_science_prefixes):
        return 'Life Science'
    elif any(naics_str.startswith(prefix) for prefix in retail_prefixes):
        return 'Retail'
    elif any(naics_str.startswith(str(code)) for code in [31, 32, 33]):
        if not any(naics_str.startswith(prefix) for prefix in it_prefixes + life_science_prefixes):
            return 'Manufacturing'
    return 'Others'

def custom_merge(reg_data, handmatch, permco_reg, permco_panel):
    # Prepare registration data
    reg_data['reg_date'] = pd.to_datetime(reg_data['reg_date'])
    reg_data['reg_year'] = reg_data['reg_date'].dt.year
    
    # Merge with handmatch data
    merged = reg_data.merge(handmatch, on='corp', how='left', indicator='merge_source')

    # Apply date filters for handmatched data
    na_mask = merged['LPERMCO'] == 'na'
    date_mask = ~na_mask & ~merged['start_fyear'].isna() & ~merged['end_fyear'].isna()
    
    if date_mask.any():
        merged.loc[date_mask, 'start_date'] = pd.to_datetime(merged.loc[date_mask, 'start_fyear'].astype(int).astype(str) + '-01-01')
        merged.loc[date_mask, 'end_date'] = pd.to_datetime(merged.loc[date_mask, 'end_fyear'].astype(int).astype(str) + '-12-31')
        date_filter = (merged['reg_date'] >= merged['start_date']) & (merged['reg_date'] <= merged['end_date'])
        keep_mask = na_mask | (date_mask & date_filter)
    else:
        keep_mask = na_mask
    
    # Process handmatched data
    merged_handmatch = merged[keep_mask].copy()
    merged_handmatch = merged_handmatch[merged_handmatch['LPERMCO'] != 'na']
    merged_handmatch['LPERMCO'] = merged_handmatch['LPERMCO'].astype('int64')
    merged_handmatch['merge_source'] = 'handmatch'

    merged_handmatch = merged_handmatch.merge(permco_panel, left_on=['LPERMCO', 'reg_year'], right_on=['LPERMCO', 'fyear'], how='left')
    merged_handmatch = merged_handmatch[merged_handmatch['active'] == 1]
    
    # Process remaining after handmatched data
    unmatched = merged[~keep_mask & (merged['merge_source'] == 'left_only')].drop(columns=[col for col in handmatch.columns if col != 'corp'])
    merged_permco = unmatched.merge(permco_reg, on='corp', how='left')
    
    # Check if registration year is among active years for LPERMCO
    merged_permco = merged_permco.merge(permco_panel, left_on=['LPERMCO', 'reg_year'], right_on=['LPERMCO', 'fyear'], how='left')
    merged_permco = merged_permco[merged_permco['active'] == 1]
    merged_permco['merge_source'] = 'permco_reg'
    
    # Combine results and clean up
    result = pd.concat([merged_handmatch, merged_permco], ignore_index=True)
    result = result.drop(['start_date', 'active','end_date', 'fyear'], axis=1, errors='ignore')
    
    return result

# Main script
if __name__ == "__main__":
    print("Processing USCO data...")
    # Load and process USCO data
    usco = pd.read_csv('predictions_usco.csv')
    usco = usco.query('not (LPERMCO == 24345 and IssuerNm == "WARNER BROS CO")')
    usco = usco[usco['prediction'] == 1]
    usco_highest_prob = usco.loc[usco.groupby(['LPERMCO', 'cluster_id_y'])['probability'].idxmax()]

    permco_reg = usco_highest_prob
    permco_reg['LPERMCO'] = permco_reg['LPERMCO'].astype(int)
    permco_reg['corp'] = permco_reg['claimant_name']

    # # Process registration data
    reg_data = pd.read_csv('registrations/combined_regs.csv')
    reg_data = reg_data.drop(reg_data.columns[0], axis=1)
    reg_data['corp'] = reg_data['claimant_name']

    # # Process handmatch data
    handmatch = pd.read_csv('training-data/usco_hand_matched.csv')
    handmatch = handmatch[['corp', 'LPERMCO', 'start_fyear', 'end_fyear']]

    # # Process CRSP and Compustat data
    print("Processing CRSP and Compustat data...")
    crsp = pd.read_csv('raw-data/crsp_link.csv', low_memory=False)
    crsp['industry'] = crsp['naics'].apply(classify_industry)
    crsp = crsp[['industry', 'gvkey', 'LPERMCO', 'LPERMNO', 'naics']]

    compustat = pd.read_csv('raw-data/compustat_all.csv', low_memory=False)
    gvkey_panel = compustat[['gvkey', 'fyear']].dropna(subset=['fyear']).drop_duplicates()
    gvkey_panel['fyear'] = gvkey_panel['fyear'].astype(int)

    # # Merge CRSP and Compustat data
    crsp_panel = pd.merge(crsp, gvkey_panel, on='gvkey', how='left').dropna(subset=['fyear']).drop_duplicates()

    # # Create PERMCO panel
    permco_panel = crsp_panel[['LPERMCO', 'fyear']].dropna(subset=['fyear']).drop_duplicates()
    permco_panel['active'] = 1

    # # Perform custom merge
    merged_regs = custom_merge(reg_data, handmatch, permco_reg, permco_panel)

    print("Processing litigation data...")
    # # Process litigation data
    litigation = pd.read_csv('predictions_litigation.csv')
    
    litigation = litigation[litigation['prediction'] == 1]
    litigation_highest_prob = litigation.loc[litigation.groupby(['LPERMCO', 'cluster_id_y'])['probability'].idxmax()]

    permco_lit = litigation_highest_prob

    # # Process litigation counts
    litdata = pd.read_csv("raw-data/cv88on.txt", sep="\t", encoding = 'iso-8859-1', low_memory = False)
    
    litdata = litdata[litdata['NOS'] == 820]   # copyright
    plt_agg = litdata.groupby(['PLT', 'TAPEYEAR']).size().reset_index(name='plt_count')
    plt_agg = plt_agg.rename(columns={'PLT': 'company', 'TAPEYEAR': 'fyear'})
    def_agg = litdata.groupby(['DEF', 'TAPEYEAR']).size().reset_index(name='def_count')
    def_agg = def_agg.rename(columns={'DEF': 'company', 'TAPEYEAR': 'fyear'})
    all_lit = pd.merge(plt_agg, def_agg, on=['company', 'fyear'], how='outer')

    # Quantify plaintiff and defensive linkage
    permcos = permco_lit[['company','LPERMCO']].drop_duplicates()
    plt_summary = pd.merge(plt_agg,permcos,on='company', how='left')
    plt_summary['LPERMCO'] = plt_summary['LPERMCO'].fillna(0).astype(int)
    plt_summary = plt_summary.groupby(plt_summary['LPERMCO'] > 0)['plt_count'].sum().reset_index()
    print(plt_summary)

    def_summary = pd.merge(def_agg,permcos,on='company', how='left')
    def_summary['LPERMCO'] = def_summary['LPERMCO'].fillna(0).astype(int)
    def_summary = def_summary.groupby(def_summary['LPERMCO'] > 0)['def_count'].sum().reset_index()
    print(def_summary)

    # # Merge litigation data
    merged_lit = pd.merge(permco_lit, all_lit, on='company', how='left')
    merged_lit['LPERMCO'] = merged_lit['LPERMCO'].fillna(0).astype(int)

    print("Merging all data...")

    # # Merge all data
    merged_regs['fyear'] = merged_regs['reg_date'].dt.year
    merged_regs = merged_regs[merged_regs['fyear'] >= 1978]
    merged_regs['LPERMCO'] = pd.to_numeric(merged_regs['LPERMCO'], errors='coerce')

    merged_regs = merged_regs.sort_values(by=['reg_num', 'corp', 'fyear', 'probability'], ascending=[True, True, True, False])
    unique_df = merged_regs.drop_duplicates(subset=['reg_num', 'corp', 'fyear'], keep='first')
    unique_df.to_csv("corp_registrations.csv", index = False)

    permco_reg_panel = unique_df.groupby(['LPERMCO', 'fyear'])['reg_num'].nunique().reset_index()
    permco_reg_panel.columns = ['LPERMCO', 'fyear', 'reg_count']
    permco_reg_panel = permco_reg_panel[permco_reg_panel['LPERMCO'] != 0]

    crsp_panel = pd.merge(crsp_panel, permco_reg_panel, on=['LPERMCO', 'fyear'], how='left')
    crsp_panel['first_year'] = crsp_panel.groupby('LPERMCO')['fyear'].transform('min')
    crsp_panel_filtered = crsp_panel[(crsp_panel['fyear'] >= 1978) & (crsp_panel['fyear'] <= 2021)]

    litigation = merged_lit[merged_lit['fyear'] <= 2021]
    litigation = litigation.sort_values(by=['company', 'fyear', 'probability'], ascending=[True, True, False])
    litigation = litigation.drop_duplicates(subset=['company', 'fyear'], keep='first')

    litigation['plt_count'] = litigation['plt_count'].fillna(0).astype(int)
    litigation['def_count'] = litigation['def_count'].fillna(0).astype(int)
    aggregated_lit = litigation.groupby(['fyear', 'LPERMCO']).agg({'plt_count': 'sum', 'def_count': 'sum'}).reset_index()

    final_data = pd.merge(crsp_panel_filtered, aggregated_lit, on=['LPERMCO', 'fyear'], how='left')
    final_data.to_csv('crsp_compustat_regdata_1978-2021_id-only.csv', index=False)

    print("Reading additional Compustat data...")
    compustat_add = pd.read_csv('raw-data/compustat_all.csv', 
                                usecols=['xad', 'xrd', 'gvkey', 'fyear', 'sale', 'ni', 'emp','rdip','acqintan','intan','at','gdwl'])

    print("Merging additional Compustat data...")
    final_data = pd.merge(final_data, compustat_add, on=['gvkey', 'fyear'], how='left')

    print("Saving final dataset...")
    final_data.to_csv('crsp_compustat_regdata_1978-2021.csv', index=False)
