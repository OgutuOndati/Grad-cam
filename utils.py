# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:05:24 2021

@author: Ranak Roy Chowdhury
"""
import warnings, pickle, torch, math, os, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import multitask_transformer_class
warnings.filterwarnings("ignore")



# loading optimized hyperparameters
def get_optimized_hyperparameters(dataset):

    path = './hyperparameters.pkl'
    with open(path, 'rb') as handle:
        all_datasets = pickle.load(handle)
        if dataset in all_datasets:
            prop = all_datasets[dataset]
    return prop
    


# loading user-specified hyperparameters
def get_user_specified_hyperparameters(args):

    prop = {}
    prop['batch'], prop['lr'], prop['nlayers'], prop['emb_size'], prop['nhead'], prop['task_rate'], prop['masking_ratio'], prop['task_type'] = \
        args.batch, args.lr, args.nlayers, args.emb_size, args.nhead, args.task_rate, args.masking_ratio, args.task_type
    return prop



# loading fixed hyperparameters
def get_fixed_hyperparameters(prop, args):
    
    prop['lamb'], prop['epochs'], prop['ratio_highest_attention'], prop['avg'] = args.lamb, args.epochs, args.ratio_highest_attention, args.avg
    prop['dropout'], prop['nhid'], prop['nhid_task'], prop['nhid_tar'], prop['dataset'] = args.dropout, args.nhid, args.nhid_task, args.nhid_tar, args.dataset
    return prop



def get_prop(args):
    
    # loading optimized hyperparameters
    # prop = get_optimized_hyperparameters(args.dataset)

    # loading user-specified hyperparameters
    prop = get_user_specified_hyperparameters(args)
    
    # loading fixed hyperparameters
    prop = get_fixed_hyperparameters(prop, args)
    return prop



def data_loader(dataset, data_path, task_type): 
    X_train = np.load(os.path.join(data_path + 'X_train.npy'), allow_pickle = True).astype(np.float64)
    X_test = np.load(os.path.join(data_path + 'X_test.npy'), allow_pickle = True).astype(np.float64)

    if task_type == 'classification':
        y_train = np.load(os.path.join(data_path + 'y_train.npy'), allow_pickle = True)
        y_test = np.load(os.path.join(data_path + 'y_test.npy'), allow_pickle = True)
    else:
        y_train = np.load(os.path.join(data_path + 'y_train.npy'), allow_pickle = True).astype(np.float64)
        y_test = np.load(os.path.join(data_path + 'y_test.npy'), allow_pickle = True).astype(np.float64)
        
    return X_train, y_train, X_test, y_test
    


def make_perfect_batch(X, num_inst, num_samples):
    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis = 0)
    return X



def mean_standardize_fit(X):
    m1 = np.mean(X, axis = 1)
    mean = np.mean(m1, axis = 0)
    
    s1 = np.std(X, axis = 1)
    std = np.mean(s1, axis = 0)
    
    return mean, std



def mean_standardize_transform(X, mean, std):
    return (X - mean) / std



def preprocess(prop, X_train, y_train, X_test, y_test):
    mean, std = mean_standardize_fit(X_train)
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]
    num_train_samples = math.ceil(num_train_inst / prop['batch']) * prop['batch']
    num_test_samples = math.ceil(num_test_inst / prop['batch']) * prop['batch']
    
    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()

    if prop['task_type'] == 'classification':
        y_train_task = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    else:
        y_train_task = torch.as_tensor(y_train).float()
        y_test = torch.as_tensor(y_test).float()
    
    return X_train_task, y_train_task, X_test, y_test



def initialize_training(prop):
    """
    Initializes the model, optimizer, and loss functions for training, including Grad-CAM frequency setup.

    Args:
        prop (dict): Dictionary of properties including model parameters and Grad-CAM frequency.

    Returns:
        model: The initialized model.
        optimizer: The initialized optimizer.
        criterion_tar: The loss function for task-aware reconstruction.
        criterion_task: The loss function for the task-specific output.
        best_model: The best model (for saving purposes).
        best_optimizer: The best optimizer (for saving purposes).
    """
    # Initialize the model
    model = multitask_transformer_class.MultitaskTransformerModel(
        prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'],
        prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'],
        prop['nhid_task'], prop['nlayers'], prop['dropout']
    ).to(prop['device'])

    # Initialize the best model
    best_model = multitask_transformer_class.MultitaskTransformerModel(
        prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'],
        prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'],
        prop['nhid_task'], prop['nlayers'], prop['dropout']
    ).to(prop['device'])

    # Define loss functions
    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss() if prop['task_type'] == 'classification' else torch.nn.MSELoss()

    # Initialize the optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=prop['lr'])

    # Optional: Set up Grad-CAM related configurations here
    # Example: Configure Grad-CAM hooks or setup based on gradcam_freq
    # If needed, create a Grad-CAM handler or store the gradcam_freq parameter for later use
    # Example:
    if 'gradcam_freq' in prop:
        gradcam_freq = prop['gradcam_freq']
        # Here you can set up a Grad-CAM handler or hook based on gradcam_freq
        # For example, initialize a list or a hook function
        # gradcam_hooks = setup_gradcam_hooks(model, gradcam_freq)

    return model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer




def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights):
    # attention_weights = attention_weights.to('cpu')
    # instance_weights = torch.sum(attention_weights, axis = 1)
    res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1])))
    index = index.cpu().data.tolist()
    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    return np.array(index2)

    

def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):
    indices = attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)
    boolean_indices = np.array([[True if i in index else False for i in range(X.shape[1])] for index in indices])
    boolean_indices_masked = np.repeat(boolean_indices[ : , : , np.newaxis], X.shape[2], axis = 2)
    boolean_indices_unmasked =  np.invert(boolean_indices_masked)
    
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = np.copy(X), np.copy(X), np.copy(X)
    X_train_tar = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_tar_masked = y_train_tar_masked[boolean_indices_masked].reshape(X.shape[0], -1)
    y_train_tar_unmasked = y_train_tar_unmasked[boolean_indices_unmasked].reshape(X.shape[0], -1)
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = torch.as_tensor(X_train_tar).float(), torch.as_tensor(y_train_tar_masked).float(), torch.as_tensor(y_train_tar_unmasked).float()

    return X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked

    

def compute_tar_loss(model, device, criterion_tar, y_train_tar_masked, y_train_tar_unmasked, batched_input_tar, \
                    batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):
    model.train()
    out_tar = model(torch.as_tensor(batched_input_tar, device = device), 'reconstruction')[0]

    out_tar_masked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_masked)].reshape(out_tar.shape[0], -1), device = device)
    out_tar_unmasked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_unmasked)].reshape(out_tar.shape[0], -1), device = device)

    loss_tar_masked = criterion_tar(out_tar_masked[ : num_inst], torch.as_tensor(y_train_tar_masked[start : start + num_inst], device = device))
    loss_tar_unmasked = criterion_tar(out_tar_unmasked[ : num_inst], torch.as_tensor(y_train_tar_unmasked[start : start + num_inst], device = device))
    
    return loss_tar_masked, loss_tar_unmasked



def compute_task_loss(nclasses, model, device, criterion_task, y_train_task, batched_input_task, task_type, num_inst, start):
    model.train()
    out_task, attn = model(torch.as_tensor(batched_input_task, device = device), task_type)
    out_task = out_task.view(-1, nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(torch.tensor(out_task.view(-1,)[ : num_inst],dtype=torch.float32), torch.as_tensor(torch.tensor(y_train_task.view(-1,)[start : start + num_inst],dtype=torch.float32), device = device)) # dtype = torch.long
    return attn, loss_task



def multitask_train(model, criterion_tar, criterion_task, optimizer, X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, \
                    y_train_task, boolean_indices_masked, boolean_indices_unmasked, prop):
    
    model.train() # Turn on the train mode
    total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task = 0.0, 0.0, 0.0
    num_batches = math.ceil(X_train_tar.shape[0] / prop['batch'])
    output, attn_arr = [], []
    
    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train_task[start : end].shape[0]
        
        optimizer.zero_grad()
        
        batched_input_tar = X_train_tar[start : end]
        batched_input_task = X_train_task[start : end]
        batched_boolean_indices_masked = boolean_indices_masked[start : end]
        batched_boolean_indices_unmasked = boolean_indices_unmasked[start : end]
        
        loss_tar_masked, loss_tar_unmasked = compute_tar_loss(model, prop['device'], criterion_tar, y_train_tar_masked, y_train_tar_unmasked, \
            batched_input_tar, batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start)
        
        attn, loss_task = compute_task_loss(prop['nclasses'], model, prop['device'], criterion_task, y_train_task, \
            batched_input_task, prop['task_type'], num_inst, start)

        total_loss_tar_masked += loss_tar_masked.item() 
        total_loss_tar_unmasked += loss_tar_unmasked.item()
        total_loss_task += loss_task.item() * num_inst
        
        # a = list(train_model.parameters())[0].clone()
        loss = prop['task_rate'] * (prop['lamb'] * loss_tar_masked + (1 - prop['lamb']) * loss_tar_unmasked) + (1 - prop['task_rate']) * loss_task
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # b = list(train_model.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))
        
        # if list(model.parameters())[0].grad is None:
        #    print("None")

        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_arr.append(torch.sum(attn, axis = 1) - torch.diagonal(attn, offset = 0, dim1 = 1, dim2 = 2))
      
    instance_weights = torch.cat(attn_arr, axis = 0)
    return total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task / y_train_task.shape[0], instance_weights



def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    if task_type == 'classification':

        loss = criterion(y_pred.view(-1,), torch.as_tensor(y.view(-1,)[:len(y_pred.view(-1))],device = device)).item()

        
        pred = y_pred.cpu().data.numpy()
        target=y.view(-1, )[:len(pred)].cpu().data.numpy()
        pred = np.argmax(pred, axis = 1)

        acc = accuracy_score(target, pred)
        prec =  precision_score(target, pred, average = avg)
        rec = recall_score(target, pred, average = avg)
        f1 = f1_score(target, pred, average = avg)
        
        results.extend([loss, acc, prec, rec, f1])
    else:
        y_pred = y_pred.squeeze().view(-1,)
        y = torch.as_tensor(y.view(-1)[:len(y_pred)], device = device)
        rmse = math.sqrt( ((y_pred - y) * (y_pred - y)).sum().data / y_pred.shape[0] )
        mae = (torch.abs(y_pred - y).sum().data / y_pred.shape[0]).item()
        results.extend([rmse, mae])
    # per_class_results = precision_recall_fscore_support(target, pred, average = None, labels = list(range(0, nclasses)))
    
    return results



def test(model, X, y, batch, nclasses, criterion, task_type, device, avg):
    model.eval() # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)
    
    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * batch)
            end = int((i + 1) * batch)
            num_inst = y[start : end].shape[0]
            
            out = model(torch.as_tensor(X[start : end], device = device), task_type)[0]
            output_arr.append(out[ : num_inst])
            y_val=y[start: start + num_inst]

    return evaluate(torch.cat(output_arr, 0), y_val, nclasses, criterion, task_type, device, avg)





# Import necessary modules
import torch
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gradcam_utils import compute_gradcam, visualize_gradcam

def training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop):
    tar_loss_masked_arr, tar_loss_unmasked_arr, tar_loss_arr, task_loss_arr, min_task_loss = [], [], [], [], math.inf
    acc, rmse, mae = 0, math.inf, math.inf

    instance_weights = torch.as_tensor(torch.rand(X_train_task.shape[0], prop['seq_len']), device=prop['device'])
    
    # Register hooks to capture gradients and activations
    def hook_fn(module, grad_in, grad_out):
        model.gradients = grad_in
        model.activations = grad_out

    target_layer = model.class_net[6]  # Example: select the layer for Grad-CAM (change as needed)
    handle = target_layer.register_forward_hook(hook_fn)
    
    for epoch in range(1, prop['epochs'] + 1):
        
        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(X_train_task, prop['masking_ratio'], prop['ratio_highest_attention'], instance_weights)
        
        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = multitask_train(
            model, criterion_tar, criterion_task, optimizer, 
            X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, y_train_task, 
            boolean_indices_masked, boolean_indices_unmasked, prop
        )
        
        tar_loss_masked_arr.append(tar_loss_masked)
        tar_loss_unmasked_arr.append(tar_loss_unmasked)
        tar_loss = tar_loss_masked + tar_loss_unmasked
        tar_loss_arr.append(tar_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', TAR Loss: ' + str(tar_loss), ', TASK Loss: ' + str(task_loss))

        # Save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())
    
        # Saved best model state at the lowest training loss is evaluated on the official test set
        test_metrics = test(best_model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task, prop['task_type'], prop['device'], prop['avg'])

        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]
        elif prop['task_type'] == 'regression' and test_metrics[0] < rmse:
            rmse = test_metrics[0]
            mae = test_metrics[1]

        # Compute and visualize Grad-CAM after each epoch (or as needed)
        if epoch % prop['gradcam_freq'] == 0:
            model.eval()
            with torch.no_grad():
                outputs, _ = model(X_test.to(prop['device']), task_type=prop['task_type'])
                loss = criterion_task(outputs, y_test.to(prop['device']))
                loss.backward()
                
                cam = compute_gradcam(model.gradients, model.activations)
                visualize_gradcam(cam, X_test[0].cpu().numpy())

    if prop['task_type'] == 'classification':
        print('Dataset: ' + prop['dataset'] + ', Acc: ' + str(acc))
    elif prop['task_type'] == 'regression':
        print('Dataset: ' + prop['dataset'] + ', RMSE: ' + str(rmse) + ', MAE: ' + str(mae))

    del model
    torch.cuda.empty_cache()
    handle.remove()  # Remove the hook when done

import torch
import numpy as np
import cv2

def compute_gradcam(gradients, activations):
    """
    Compute the Grad-CAM heatmap from the captured gradients and activations.
    """
    gradients = gradients[0].mean(dim=[0, 2])
    activations = activations[0]

    # Compute the weighted sum of activations
    weights = gradients
    cam = torch.zeros(activations.shape[1:])

    for i, w in enumerate(weights):
        cam += w * activations[i]

    # Apply ReLU activation
    cam = torch.nn.functional.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()  # Normalize

    return cam.numpy()