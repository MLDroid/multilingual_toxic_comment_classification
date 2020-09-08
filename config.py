import psutil
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_validation_set_fname():
    if INFERENCE_TYPE == 'en':
        valid_fname = 'en-to-en-small-valid.csv'
    elif INFERENCE_TYPE == 'ml':
        valid_fname = 'ml-small-valid.csv'
    else:
        print(f'Invalid inference type: {INFERENCE_TYPE}')
        exit(-1)
    return valid_fname


def get_train_set_fname():
    if TRAINING_TYPE == 'en':
        train_fname = 'en-to-en-small-train.csv'
    elif INFERENCE_TYPE == 'ml':
        train_fname = 'en-to-en-small-and-trans-train.csv'
    else:
        print(f'Invalid inference type: {INFERENCE_TYPE}')
        exit(-1)
    return train_fname




MODEL_NAME =  sys.argv[1] #'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'albert-base-v2'
BATCH_SIZE = int(sys.argv[2]) #just checking if 300 is OK in multiGPU setting
LR = float(sys.argv[3])
TRAINING_TYPE = sys.argv[4].lower() #can be ml or en
INFERENCE_TYPE = sys.argv[5].lower() #can be ml or en
# MODEL_NAME = 'xlm-roberta-base' #'bert-base-multilingual-uncased'
# BATCH_SIZE = 100
# LR = 5e-4
# INFERENCE_TYPE = 'ml' #or 'en'

data_folder = './data'
train_fname = get_train_set_fname()
train_fname = os.path.join(data_folder, train_fname)

valid_fname = get_validation_set_fname()
valid_fname = os.path.join(data_folder, valid_fname)

IS_LOWER = True if 'uncased' in MODEL_NAME else False

MAX_SEQ_LEN = 4096 if 'longformer' in MODEL_NAME else 512
NUM_EPOCHS = 20
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
SAVE_EVERY = 5

BERT_LAYER_FREEZE = bool(int(sys.argv[6]))

MULTIGPU = True if torch.cuda.device_count() > 1 else False #when using xlarge vs 16x large AWS m/c

#these 2 are not used yet
TRAINED_MODEL_FNAME_PREFIX = INFERENCE_TYPE.upper() + '_'+ MODEL_NAME.upper()+'_model'
TRAINED_MODEL_FNAME = None #MODEL_NAME.upper()+'_toxic_comment_model_e_10.pt'

START_TRAINING_EPOCH_AT = 1

CONTEXT_VECTOR_SIZE = 1024 if 'large' in MODEL_NAME else 768
PREDICTIONS_FNAME = '_'.join([MODEL_NAME.upper(), 'tr', TRAINING_TYPE.upper(),
                              'inf',INFERENCE_TYPE.upper(),'preds.txt'])




