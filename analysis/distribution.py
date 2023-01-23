import os
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
from functools import reduce
from torchvision import transforms as T

def analyse(df):
    interest_rows = df.loc[((df['bicam_btarget'] != df['bicam_bpred']) \
                            & (df['bicam_ntarget'] != df['bicam_npred'])) \
                            & (df['btarget'] != df['bpred'])
                            & (df['ntarget'] != df['npred'])]
    interest_rows.to_csv('./analyse.csv')
    dir = "/home/chandramouli/Documents/kaggle/CIFAR-100/test/"
    files= []
    for i in range(interest_rows.shape[0]):
        path = interest_rows.iloc[i]['path']
        f = glob(os.path.join(dir, f"*/{path}"))
        files.append(f[0])
    text = "\n".join(files)
    with open("./distribution_images_nfbf-bicamnfbf.txt", "w") as f:
        f.write(text)
    f.close()

def statistical_analysis(df):
    total = df.shape[0]
    print(total)
    interest_rows = df.loc[((df['bicam_btarget'] == df['bicam_bpred']) \
                            & (df['bicam_ntarget'] == df['bicam_npred'])) \
                            & (df['btarget'] == df['bpred'])
                            & (df['ntarget'] == df['npred'])]
    print(interest_rows.shape[0])

def read_write(args):
    bicam_bdata = np.loadtxt(f'{args.bicam_broad}',
                dtype={'names': ('path', 'bicam_btarget', 'bicam_bpred'),
                'formats': ('|U231', np.int32, np.int32)},
                delimiter=',', skiprows=0)
    bicam_ndata = np.loadtxt(f'{args.bicam_narrow}',
                dtype={'names': ('path', 'bicam_ntarget', 'bicam_npred'),
                'formats': ('|U231', np.int32, np.int32)},
                delimiter=',', skiprows=0)
    bdata = np.loadtxt(f'{args.broad}',
                dtype={'names': ('path', 'btarget', 'bpred'),
                'formats': ('|U231', np.int32, np.int32)},
                delimiter=',', skiprows=0)
    ndata = np.loadtxt(f'{args.narrow}',
                dtype={'names': ('path', 'ntarget', 'npred'),
                'formats': ('|U231', np.int32, np.int32)},
                delimiter=',', skiprows=0)
    for d in bicam_bdata:
        d[0] = d[0].rsplit('/', 1)[1]
    for d in bicam_ndata:
        d[0] = d[0].rsplit('/', 1)[1]
    for d in bdata:
        d[0] = d[0].rsplit('/', 1)[1]
    for d in ndata:
        d[0] = d[0].rsplit('/', 1)[1]
    bicam_bdata_df = pd.DataFrame(bicam_bdata, 
                                        columns=['path', 'bicam_btarget', 'bicam_bpred'])
    bicam_ndata_df = pd.DataFrame(bicam_ndata, 
                                        columns=['path', 'bicam_ntarget', 'bicam_npred'])
    bdata_df = pd.DataFrame(bdata, columns=['path', 'btarget', 'bpred'])
    ndata_df = pd.DataFrame(ndata, columns=['path', 'ntarget', 'npred'])
    dfs = [bicam_bdata_df, bicam_ndata_df, bdata_df, ndata_df]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='path'), dfs)
    df_final.dropna(inplace=True)
    return df_final

def main(args):
    df_final = read_write(args)
    analyse(df_final)
    # statistical_analysis(df_final)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bicam-broad", type=str, 
                        default="./bicameral_broad.txt",)
    parser.add_argument("--bicam-narrow", type=str, 
                        default="./bicameral_narrow.txt",)
    parser.add_argument("--broad", type=str, 
                        default="./broad.txt",)
    parser.add_argument("--narrow", type=str, 
                        default="./narrow.txt",)
    args = parser.parse_args()
    main(args)