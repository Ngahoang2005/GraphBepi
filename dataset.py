import os
import esm
import pickle as pk
import torch
import re
import numpy as np
import torch
import warnings
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader,Dataset
warnings.simplefilter('ignore')
class PDB(Dataset):
    def __init__(
        self,mode='train',fold=-1,root='./data/BCE_633',self_cycle=False
    ):
        self.root=root
        assert mode in ['train','val','test']
        if mode in ['train','val']:
            with open(f'{self.root}/train.pkl','rb') as f:
                self.samples=pk.load(f)
        else:
            with open(f'{self.root}/test.pkl','rb') as f:
                self.samples=pk.load(f)
        self.data=[]
        idx=np.load(f'{self.root}/cross-validation.npy')
        cv=10
        inter=len(idx)//cv
        ex=len(idx)%cv
        if mode=='train':
            order=[]
            for i in range(cv):
                if i==fold:
                    continue
                order+=list(idx[i*inter:(i+1)*inter+ex*(i==cv-1)])
        elif mode=='val':
            order=list(idx[fold*inter:(fold+1)*inter+ex*(fold==cv-1)])
        else:
            order=list(range(len(self.samples)))
        order.sort()
        tbar=tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root,self_cycle)
            self.data.append(self.samples[i])
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        seq=self.data[idx]
        feat=torch.cat([seq.feat,seq.dssp],1)
        return {
            'feat':feat,
            'label':seq.label,
            'adj':seq.adj,
            'edge':seq.edge,
        }
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/BCE_633', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    args = parser.parse_args()
    root = args.root
    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    
    os.system(f'cd {root} && mkdir PDB purePDB feat dssp graph')
    # model=None
    model,_=esm.pretrained.esm2_t36_3B_UR50D()
    model=model.to(device)
    model.eval()
    train='total.csv'
    initial(train,root,model,device)
    with open(f'{root}/total.pkl','rb') as f:
        dataset=pk.load(f)
    dates={i.name:i.date for i in dataset}
#     with open(f'{root}/date.pkl','rb') as f:
#         dates=pk.load(f)
    filt_data=[]
    for i in dataset:
        if len(i)<1024 and i.label.sum()>0:
            filt_data.append(i)
    month={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    TEST_CUTOFF = 20210401  # yyyymmdd
    DATES_FOR_CV = [] # Khởi tạo danh sách lưu ngày tháng cho Cross Validation
    trainset, testset = [], []

    for it in filt_data:
        # Lấy ngày tháng thô từ dictionary 'dates'
        raw = str(dates[it.name]).strip()
        
        # Chuẩn hóa phân tách bằng Regex (tách theo dấu -, /, hoặc khoảng trắng)
        parts = re.split(r'[-/\s]+', raw)
        
        try:
            # Trường hợp 1: Dạng 'DD-MON-YY' hoặc 'DD-MON-YYYY' (VD: 11-FEB-21)
            if len(parts) >= 3 and parts[1].isalpha():
                d = int(parts[0])
                m = month[parts[1].upper()[:3]] # Lấy 3 chữ cái đầu của tháng
                y_str = parts[2]
                if len(y_str) == 4:
                    y = int(y_str)
                else:
                    y2 = int(y_str)
                    # Quy ước: năm < 23 hiểu là 20xx, từ 23 trở đi hiểu là 19xx
                    y = 2000 + y2 if y2 < 23 else 1900 + y2
                    
            # Trường hợp 2: Dạng 'YYYY-MM-DD' (VD: 2021-04-05)
            elif len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 4:
                y = int(parts[0])
                m = int(parts[1])
                d = int(parts[2])
                
            else:
                print(f"[WARN] Unrecognized date '{raw}' for {it.name}; skipping")
                continue
                
            # Tính toán ra số int dạng YYYYMMDD để so sánh
            date_int = y * 10000 + m * 100 + d
            
        except Exception as e:
            print(f"[WARN] Bad date '{raw}' for {it.name}: {e}; skipping")
            continue
            
        # Chia tập train/test dựa vào mốc TEST_CUTOFF
        if date_int < TEST_CUTOFF:
            DATES_FOR_CV.append(date_int)
            trainset.append(it)
        else:
            testset.append(it)
    with open(f'{root}/train.pkl','wb') as f:
        pk.dump(trainset,f)
    with open(f'{root}/test.pkl','wb') as f:
        pk.dump(testset,f)
    idx=np.array(DATES_FOR_CV).argsort()
    np.save(f'{root}/cross-validation.npy',idx)