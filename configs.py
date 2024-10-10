# about dataset
data_name = 'DynHCP'
sub_data_type = 'Gender'
split_type = 'by_ind'
net_type = 'group'

# about model
conv = 'sage'
read_out_type = 'mean'
num_edges = 10
hidden_feats = 128
out_feats = 128
use_init_struc = False
with_contrast = True
structure_learning = True
cuda = 'cuda:0'

# about training
lr = 0.001
epochs = 100
batch_size = 32
patience_threashold = 50