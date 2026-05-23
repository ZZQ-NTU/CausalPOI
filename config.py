lm_names = {'multi_bert': 'bert-base-multilingual-cased', 'bert': 'bert-base-uncased', 'distilbert': 'distilbert-base-uncased', 'roberta': 'roberta-base'}
lm_hidden_sizes = {'multi_bert': 768, 'bert': 768, 'distilbert': 768, 'roberta': 768}
default_lm = 'bert'
device = 'cuda'
lm = 'bert'
no_context = 'context: none'
log_file_name = 'log_file.txt'

MAX_DIST = 1000
epochs = 10
lr = 3e-5
dropout = 0.2
bs = 32
sep_width = 50
