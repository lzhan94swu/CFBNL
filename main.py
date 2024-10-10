import os
import gc
import argparse

from torch import _scaled_dot_product_attention_math

from models import *
from gnn_model import *
from utils import *
import configs

import time


def main(args):
    # about dataset
    data_name = args.data_name
    if data_name == 'DynHCP':
        sub_data_type = args.sub_data_type
    else:
        sub_data_type = None
    split_type = args.split_type
    if split_type == 'loocv':
        test_ind = args.test_ind
    net_type = args.net_type

    # about model
    conv = args.conv
    read_out_type = args.read_out_type
    num_edges = args.num_edges
    hidden_feats = args.hidden_feats
    out_feats = args.out_feats
    use_init_struc = args.use_init_struc
    if use_init_struc:
        net_type = 'sample'
    with_contrast = args.with_contrast
    cuda_device = args.cuda
    structure_learning = args.structure_learning

    if with_contrast == True:
        contrast = 'contrast'
    else:
        contrast = 'no_contrast'

    # about training
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    patience_threashold = args.patience_threashold

    # Set device
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    init_strucs, graph_feats, all_nets_index, ind_labels, state_labels = read_graph_matrices(data_name, sub_type=sub_data_type)
    num_nodes = graph_feats[0].shape[0]
    if use_init_struc == True:
        num_edges = int(np.array([np.count_nonzero(init_struc) for init_struc in init_strucs], dtype=np.int32).mean())
    num_cls = len(np.unique(state_labels))
    num_inds = len(np.unique(ind_labels))
    num_samples = len(graph_feats)
    in_feats = graph_feats[0].shape[1]

    # Split data into train, validation, and test sets
    split_type_dict = {'by_ind': data_set_split_by_ind, 'across_ind': data_set_split_across_ind}
    [train_index, val_index, test_index,
    train_ind_labels, val_ind_labels, test_ind_labels,
    train_state_labels, val_state_labels, test_state_labels] = split_type_dict[split_type](all_nets_index, ind_labels, state_labels, rate=0.2, random_state=42)

    train_feats = graph_feats[train_index]
    val_feats = graph_feats[val_index]
    test_feats = graph_feats[test_index]

    # Create your GCN model
    gnn = GNNModel(in_feats, hidden_feats, out_feats, num_nodes, conv, read_out_type).to(device)

    # Create your contrastive learning model
    model = CGSL(gnn,
                   out_feats,
                   num_nodes,
                   num_edges,
                   num_cls=num_cls,
                   num_inds=num_inds,
                   num_samples=num_samples,
                   net_type=net_type,
                   use_init_struc=use_init_struc,
                   init_struc=init_strucs,
                   structure_learning=structure_learning,
                   batch_size=batch_size).to(device)

    # Create your optimizer for contrastive learning
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Create the graph dataset loaders for train, validation, and test sets
    train_loader = graph_dataset_loader(train_feats, train_index, train_ind_labels, train_state_labels, batch_size)
    val_loader = graph_dataset_loader(val_feats, val_index, val_ind_labels, val_state_labels, batch_size)
    test_loader = graph_dataset_loader(test_feats, test_index, test_ind_labels, test_state_labels, batch_size)

    best_val_accuracy = 0.0
    best_model_state_dict = None
    patience = 0

    # Training loop for contrastive learning
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            # Move batch to device
            batch = [item.to(device) for item in batch]

            # Unpack batch
            network_matrices, nets_index, ind_labels, state_labels = batch

            net_ind_dict = {'ind': ind_labels, 'cls': state_labels, 'sample': nets_index, 'group': None}
            net_index = net_ind_dict[net_type]
            # Forward pass
            logits, embeddings, _ = model(network_matrices, net_index)
            # logits = torch.argmax(logits, dim=1)

            # Compute multi-classification loss
            loss1 = F.cross_entropy(logits, state_labels.long())
            if with_contrast == True:
                loss2 = contrastive_loss(embeddings, ind_labels)
                loss = loss1 + loss2
            else:
                loss = loss1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            # print(f"Multi-Classification - Epoch: {epoch+1}, Loss: {loss2.item()}")
        # Evaluation on validation set
        model.eval()
        correct = 0
        total = 0
        val_losses = []
        for batch in val_loader:
            # Move batch to device
            batch = [item.to(device) for item in batch]

            # Unpack batch
            network_matrices, nets_index, ind_labels, state_labels = batch

            net_ind_dict = {'ind': ind_labels, 'cls': state_labels, 'sample': nets_index, 'group': None}
            net_index = net_ind_dict[net_type]

            # Forward pass
            logits, _, _ = model(network_matrices, net_index)

            # Compute multi-classification loss
            loss = F.cross_entropy(logits, state_labels.long())
            val_losses.append(loss.item())

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += state_labels.size(0)
            correct += (predicted == state_labels).sum().item()

        avg_val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = correct / total
        # print(f"Average Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}")

        # Early stopping based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()
            # Specify the directory path to save the model
            state_save_dir = "./{}_uni_weight/".format(contrast)
            os.makedirs(state_save_dir, exist_ok=True)

            # Save the best_model_state_dict to the specified directory
            torch.save(best_model_state_dict, state_save_dir + f"best_model_state_dict_{data_name}_{sub_data_type}_{split_type}_{net_type}_{conv}.pth")
            patience = 0

        if patience > patience_threashold:
            break
        else:
            patience += 1

        # Evaluation on test set
        model.eval()
        correct = 0
        total = 0
        test_losses = []
        for batch in test_loader:
            # Move batch to device
            batch = [item.to(device) for item in batch]

            # Unpack batch
            network_matrices, nets_index, ind_labels, state_labels = batch

            net_ind_dict = {'ind': ind_labels, 'cls': state_labels, 'sample': nets_index, 'group': None}
            net_index = net_ind_dict[net_type]

            # Forward pass
            logits, _, _ = model(network_matrices, net_index)

            # Compute multi-classification loss
            loss = F.cross_entropy(logits, state_labels.long())
            test_losses.append(loss.item())

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += state_labels.size(0)
            correct += (predicted == state_labels).sum().item()

        avg_test_loss = sum(test_losses) / len(test_losses)
        test_accuracy = correct / total
        # print(f"Average Test Loss: {avg_test_loss}, Accuracy: {test_accuracy}")

    # Evaluation on test set
    best_model_state_dict = torch.load(state_save_dir + f"best_model_state_dict_{data_name}_{sub_data_type}_{split_type}_{net_type}_{conv}.pth")
    model.load_state_dict(best_model_state_dict)
    model.eval()
    correct = 0
    total = 0
    test_losses = []
    for batch in test_loader:
        # Move batch to device
        batch = [item.to(device) for item in batch]

        # Unpack batch
        network_matrices, nets_index, ind_labels, state_labels = batch

        net_ind_dict = {'ind': ind_labels, 'cls': state_labels, 'sample': nets_index, 'group': None}
        net_index = net_ind_dict[net_type]

        # Forward pass
        logits, _, _ = model(network_matrices, net_index)

        # Compute multi-classification loss
        loss = F.cross_entropy(logits, state_labels.long())
        test_losses.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += state_labels.size(0)
        correct += (predicted == state_labels).sum().item()

    avg_test_loss = sum(test_losses) / len(test_losses)
    Final_accuracy = correct / total
    print(f"Final Test Loss: {avg_test_loss}, Accuracy: {Final_accuracy}")
    return Final_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper-Parameters')
    parser.add_argument('--data_name', type=str, default=configs.data_name, choices=['cog_state', 'slim', 'DynHCP'], help='The name of the selected dataset')
    parser.add_argument('--sub_data_type', type=str, default=configs.sub_data_type, choices=['Age', 'Gender', 'Activity'], help='When choose bciv as the dataset, you can choose the sub dataset type')
    parser.add_argument('--split_type', type=str, default=configs.split_type, choices=['by_ind', 'across_ind'], help='Choose the split type')
    parser.add_argument('--net_type', type=str, default=configs.net_type, choices=['group', 'cls', 'ind', 'samples'], help='The expected type of the learned network')

    parser.add_argument('--conv', type=str, default=configs.conv, choices=['gcn', 'sage', 'gat', 'gin'], help='The type of the graph convolution layer')
    parser.add_argument('--read_out_type', type=str, default=configs.read_out_type, choices=['mean', 'max', 'add', 'dense', 'trans'], help='The type of the readout layer')
    parser.add_argument('--num_edges', type=int, default=configs.num_edges, help='The number of edges to keep in the network')
    parser.add_argument('--hidden_feats', type=int, default=configs.hidden_feats, help='The dimension of the hidden layer')
    parser.add_argument('--out_feats', type=int, default=configs.out_feats, help='The dimension of the output layer')
    parser.add_argument('--use_init_struc', type=bool, default=configs.use_init_struc, help='Whether to use the initial network structure')
    parser.add_argument('--with_contrast', type=bool, default=configs.with_contrast, help='Whether to use contrastive learning')
    parser.add_argument('--structure_learning', type=bool, default=configs.structure_learning, help='Whether to use structure learning')
    parser.add_argument('--cuda', type=str, default=configs.cuda, help='Which device to be used')

    parser.add_argument('--lr', type=float, default=configs.lr, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=configs.epochs, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=configs.batch_size, help='The batch size for training')
    parser.add_argument('--patience_threashold', type=int, default=configs.patience_threashold, help='The patience threshold for early stopping')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()

    print(args)
    accs = []
    for _ in range(5):
        acc = main(args)
        accs.append(acc)
    print(f'Average accuracy: {np.mean(accs)}, std: {np.std(accs)}')
    with open(f'./{args.data_name}_{args.sub_data_type}_{args.split_type}_{args.net_type}_{args.conv}.txt', 'w') as f:
        f.write(f'Average accuracy: {np.mean(accs)}, std: {np.std(accs)}')