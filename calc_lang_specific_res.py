import pandas as pd, os, sys
from sklearn.metrics import roc_auc_score, f1_score


res_fname = sys.argv[1]
all_y_pred = [int(l.strip()) for l in open(res_fname)]
print(f'Loaded {len(all_y_pred)} predictions from {res_fname}')


valid_fname = sys.argv[2]
df = pd.read_csv(valid_fname)
print(f'Loaded df of shape: {df.shape} from {valid_fname}')
langs = sorted(df['lang'].unique())
all_y_true = df['toxic']
lang_to_inds = {}
for l in langs:
    ldf = df[df.lang == l]
    start_index = min(ldf.index)
    end_index = max(ldf.index)
    lang_to_inds[l] = (start_index,end_index)

overall_auc = round(roc_auc_score(all_y_true, all_y_pred)*100, 2)
overall_f1 = round(f1_score(all_y_true, all_y_pred)*100, 2)
print(f'Overall auc: {overall_auc} and f1: {overall_f1}')

for l,(start_index,end_index) in lang_to_inds.items():
    y_true = all_y_true[start_index: end_index+1]
    y_pred = all_y_pred[start_index: end_index+1]
    f1 = round(f1_score(y_true, y_pred)*100,2)
    auc = round(roc_auc_score(y_true, y_pred)*100, 2)
    print(f'For lang: {l}, auc: {auc} and f1: {f1}')




