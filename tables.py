import pandas as pd
import numpy as np
import string

def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def abbreviate_name(name, duplicate_col_check, duplicate_counter):
    vowels = "aeiouAEIOU0123456789"
    sct = "".join([char for char in name if char not in vowels])
    sct = sct.strip().replace(" ", "")
    sct = remove_punctuation(sct)
    abbr_col = sct[:3]  

    if abbr_col in duplicate_col_check:
        abbr_col = f"{abbr_col}{duplicate_counter}"
        duplicate_counter += 1

    duplicate_col_check.append(abbr_col)

    return abbr_col, duplicate_counter

def adjust_percentages(df):
    for col in df.columns:
        total_percent = df[col].sum()
        if total_percent != 100:
            max_idx = df[col].idxmax()
            df.at[max_idx, col] += 100 - total_percent
    return df

def process_dataframe(data1, fixed_column='Case_Control_new'):
    columns_to_remove = ['patient_id', 'last_updated']
    for col in columns_to_remove:
        if col in data1.columns:
            data1 = data1.drop(columns=[col])

    for col in data1.columns:
        if data1[col].dtype.name == 'category':
            if 'Missing' not in data1[col].cat.categories:
                data1[col] = data1[col].cat.add_categories('Missing')
    
    data1.fillna('Missing', inplace=True)
    
    if fixed_column not in data1.columns:
        data1[fixed_column] = 'case'
    
    columns = data1.columns.tolist()
    if fixed_column in columns:
        columns.remove(fixed_column)

    user_population_at_risk = {}

    '''
    Assigning random population remove this when deploying
    '''
    def assign_random_population(column_name):
        unique_values = data1[column_name].dropna().unique()
        if 'Missing' in unique_values:
            unique_values = unique_values[unique_values != 'Missing']
        for value in unique_values:
            population = np.random.randint(1, 101)
            user_population_at_risk[(column_name, value)] = population

    # Assign random values for population at risk for each categorical column
    for column in data1.columns:
        if column != fixed_column:
            assign_random_population(column)

    final_result = []
    freq_table_list = []
    attack_rate_list = []

    # Track duplicate column abbreviations
    duplicate_col_check = []
    duplicate_counter = 1

    for column in columns:
        abbr_column, duplicate_counter = abbreviate_name(column, duplicate_col_check, duplicate_counter)

        crosstab_freq = pd.crosstab(data1[column], data1[fixed_column], margins=False)
        crosstab_perc = pd.crosstab(data1[column], data1[fixed_column], normalize='columns') * 100
        crosstab_perc = crosstab_perc.applymap(lambda x: int(x) if x < 0.5 else round(x))
        crosstab_perc = adjust_percentages(crosstab_perc)
        
        total_freq = crosstab_freq.sum(axis=1)

        if 'case' not in crosstab_freq.columns:
            crosstab_freq['case'] = 0
            crosstab_perc['case'] = 0

        if 'control' not in crosstab_freq.columns:
            crosstab_freq['control'] = 0
            crosstab_perc['control'] = 0

        result_dict = {
            'Category': crosstab_freq.index,
            'Total (N)': total_freq.values,
            'Case (n)': crosstab_freq['case'].values,
            'Case (%)': crosstab_perc['case'].values,
            'Control (n)': crosstab_freq['control'].values,
            'Control (%)': crosstab_perc['control'].values,
        }

        result_df = pd.DataFrame(result_dict)
        result_df = result_df.reset_index(drop=True)
        result_df.index = pd.MultiIndex.from_product([[abbr_column], result_df.index])

        final_result.append(result_df)

        freq_df = result_df[['Category', 'Case (n)', 'Case (%)']].copy()
        freq_df = freq_df.rename(columns={'Case (n)': 'Cases (n)', 'Case (%)': 'Case percent (%)'})
        freq_df['Variables'] = abbr_column
        freq_table_list.append(freq_df)

        cases_n = []
        population_at_risk = []
        attack_rate = []
        case_percent = []

        for idx in result_df.index.get_level_values(1):
            cat = result_df.loc[(abbr_column, idx), 'Category']
            covid_n = result_df.loc[(abbr_column, idx), 'Case (n)']
            population_n = user_population_at_risk.get((column, cat), 0)
            attack_rate_value = (covid_n / population_n) * 100 if population_n != 0 else 0

            cases_n.append(covid_n)
            population_at_risk.append(population_n)
            attack_rate.append(round(attack_rate_value, 0))
            case_percent.append(result_df.loc[(abbr_column, idx), 'Case (%)'])

        attack_df = pd.DataFrame({
            'Category': result_df['Category'],
            'Cases (n)': cases_n,
            'Population at risk (N)': population_at_risk,
            'Attack rate (%)': attack_rate,
            'Case percent (%)': case_percent
        })

        attack_df['Variables'] = abbr_column
        attack_df = attack_df[['Variables', 'Category', 'Cases (n)', 'Population at risk (N)', 'Attack rate (%)', 'Case percent (%)']]
        attack_rate_list.append(attack_df)

    crosstab_table = pd.concat(final_result).reset_index()
    crosstab_table = crosstab_table[['level_0', 'Category', 'Total (N)', 'Case (n)', 'Case (%)', 'Control (n)', 'Control (%)']]
    crosstab_table.columns = ['Variables', 'Category', 'Total (N)', 'Case (n)', 'Case (%)', 'Control (n)', 'Control (%)']

    freq_table = pd.concat(freq_table_list).reset_index(drop=True)
    freq_table = freq_table[['Variables', 'Category', 'Cases (n)', 'Case percent (%)']]

    attack_rate_table = pd.concat(attack_rate_list).reset_index(drop=True)

    return crosstab_table, freq_table, attack_rate_table



