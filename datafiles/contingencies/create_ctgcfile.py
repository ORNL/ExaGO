input_file = r'case_ACTIVSg70k.m'
output_folder = r'out'

import os
import re
import pandas as pd

def extract_matrix_with_headers(text, key):
    pattern = rf"(?:%%[^\r\n]*[\r\n])?%[ \t]*([^\r\n]*)[\r\n]\s*mpc\.{key}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, re.DOTALL)
    header_line = match.group(1).strip()
    raw_headers = re.split(r'[ \t]+', header_line)
    headers = raw_headers if any(not re.fullmatch(r'\d+', h) for h in raw_headers) else None
    block = match.group(2)
    lines = [line.strip().rstrip(';') for line in block.strip().split('\n') if line.strip()]
    data = [list(map(float, re.split(r'[ \t]+', line))) for line in lines]
    df = pd.DataFrame(data)
    if headers and len(headers) == df.shape[1]:
        df.columns = headers
    return df

def process_m_with_headers_df(m_file_path):
    with open(m_file_path, 'r') as f:
        content = f.read()
    summary_index = content.find('mpc.summary')
    if summary_index != -1:
        content = content[:summary_index]
    dfs = {}
    for key in ['bus', 'gen', 'branch']:
        df = extract_matrix_with_headers(content, key)
        output_path = os.path.join(output_folder, f"{key}.csv")
        #df.to_csv(output_path, index=False)
        dfs[key] = df
    df_load = dfs.get('bus')
    df_gen = dfs.get('gen')
    df_branch = dfs.get('branch')
    return df_load, df_gen, df_branch

df_load_initial, df_gen_initial, df_branch_initial = process_m_with_headers_df(input_file)

df_load = df_load_initial[df_load_initial['Pd'] > 0].reset_index(drop=True)
n_loads = df_load.shape[0]
df_load = df_load.sort_values('bus_i').reset_index(drop=True)
df_load['id_or_circuit'] = df_load.groupby('bus_i').cumcount() + 1

df_gen = df_gen_initial[df_gen_initial['status'] == 1].reset_index(drop=True)
n_gen = df_gen.shape[0]
df_gen = df_gen.sort_values('bus').reset_index(drop=True)
df_gen['id_or_circuit'] = df_gen.groupby('bus').cumcount() + 1

df_branch = df_branch_initial[(df_branch_initial['status'] == 1) & (df_branch_initial['ratio'] == 0)].reset_index(drop=True)
n_branch = df_branch.shape[0]
df_branch = df_branch.sort_values(['fbus', 'tbus']).reset_index(drop=True)
df_branch['id_or_circuit'] = df_branch.groupby(['fbus', 'tbus']).cumcount() + 1

df_tr = df_branch_initial[(df_branch_initial['status'] == 1) & (df_branch_initial['ratio'] > 0)].reset_index(drop=True)
n_tr = df_tr.shape[0]
df_tr = df_tr.sort_values(['fbus', 'tbus']).reset_index(drop=True)
df_tr['id_or_circuit'] = df_tr.groupby(['fbus', 'tbus']).cumcount() + 1

def create_contingency_df(df, df_type):
    cont_df = pd.DataFrame()
    cont_df['Num'] = df.index + 1  # contingency numbers starting from 1
    type_mapping = {'gen': 1, 'branch': 2, 'tr': 3, 'load': 4}
    cont_df['Type'] = type_mapping[df_type]
    if df_type == 'gen':
        cont_df['Bus'] = df['bus']
    elif df_type == 'load':
        cont_df['Bus'] = df['bus_i']
    else:  # branch or transformer
        cont_df['Bus'] = 0
    if df_type in ['branch', 'tr']:
        cont_df['Fbus'] = df['fbus']
    else:
        cont_df['Fbus'] = 0
    if df_type in ['branch', 'tr']:
        cont_df['Tbus'] = df['tbus']
    else:
        cont_df['Tbus'] = 0
    cont_df['Id'] = df['id_or_circuit']
    cont_df['Status'] = 0
    cont_df['Prob'] = 0.001
    return cont_df

df_gen_cont = create_contingency_df(df_gen, 'gen')
df_load_cont = create_contingency_df(df_load, 'load')
df_branch_cont = create_contingency_df(df_branch, 'branch')
df_tr_cont = create_contingency_df(df_tr, 'tr')

df_all = pd.concat([df_gen_cont, df_load_cont, df_branch_cont, df_tr_cont], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)  # random_state for reproducibility
df_all['Num'] = df_all.index + 1  # start from 1 if you like
#df_all.to_csv('cont.csv', header=False, index=False)

columns_order = ['Num', 'Type', 'Bus', 'Fbus', 'Tbus', 'Id', 'Status', 'Prob']
df_all['Num'] = df_all['Num'].astype(int)
df_all['Type'] = df_all['Type'].astype(int)
df_all['Bus'] = df_all['Bus'].astype(int)
df_all['Fbus'] = df_all['Fbus'].astype(int)
df_all['Tbus'] = df_all['Tbus'].astype(int)
df_all['Id'] = df_all['Id'].astype(str)
df_all['Status'] = df_all['Status'].astype(int)
df_all['Prob'] = df_all['Prob'].astype(float)
df_txt = df_all[columns_order]

df_txt.to_csv('70thousandbus.cont', sep=',', header=False, index=False, float_format='%.3f', lineterminator='\n')

with open('70thousandbus.cont', 'r') as f:
    lines = f.readlines()
with open('70thousandbus.cont', 'w') as f:
    for line in lines:
        parts = [p.strip() for p in line.strip().split(',')]
        padded = []
        for i, p in enumerate(parts):
            if i in [0,1,2,3,4,6]:   # integer columns
                padded.append(f'{int(p):>5}')
            elif i == 5:              # string column
                padded.append(f"'{p}'")
            else:                     # float column
                padded.append(f'{float(p):>6.3f}')
        f.write(', '.join(padded) + '\n')