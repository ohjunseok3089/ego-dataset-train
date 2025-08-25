#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import torch
import numpy as np


def adaptive_f1(preds, labels_hm, labels, dataset):
    """
    Automatically select the threshold getting the best f1 score.
    """
    # Numpy
    # # thresholds = np.linspace(0, 1.0, 51)
    # thresholds = np.linspace(0, 0.2, 11)
    # # thresholds = np.array([0.5])
    # preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    # all_preds = np.zeros(shape=(thresholds.shape + labels.shape))
    # all_labels = np.zeros(shape=(thresholds.shape + labels.shape))
    # binary_labels = (labels > 0.001).astype(np.int)
    # for i in range(thresholds.shape[0]):
    #     binary_preds = (preds.squeeze(1) > thresholds[i]).astype(np.int)
    #     all_preds[i, ...] = binary_preds
    #     all_labels[i, ...] = binary_labels
    # tp = (all_preds * all_labels).sum(axis=(3, 4))
    # fg_labels = all_labels.sum(axis=(3, 4))
    # fg_preds = all_preds.sum(axis=(3, 4))
    # recall = (tp / (fg_labels + 1e-6)).mean(axis=(1, 2))
    # precision = (tp / (fg_preds + 1e-6)).mean(axis=(1, 2))
    # f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    # max_idx = np.argmax(f1)
    # return f1[max_idx], recall[max_idx], precision[max_idx], thresholds[max_idx]

    # PyTorch (To speed up calculation, use different search space for different datasets)
    if 'forecast' in dataset and 'aria' not in dataset:  # gaze forecasting on ego4d dataset
        thresholds = np.linspace(0.01, 0.07, 31)
        # thresholds = np.linspace(0, 1.0, 21)
    elif 'forecast' in dataset and 'aria' in dataset:  # gaze forecasting on aria dataset
        thresholds = np.linspace(0.0, 0.02, 21)
        # thresholds = np.linspace(0, 1.0, 21)
    else:  # gaze estimation
        thresholds = np.linspace(0, 0.02, 11)
        # thresholds = np.linspace(0, 1.0, 21)

    all_preds = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    all_labels = torch.zeros(size=(thresholds.shape + labels_hm.size()), device=labels_hm.device)
    binary_labels = (labels_hm > 0.001).int()  # change to 0.001
    for i in range(thresholds.shape[0]):
        binary_preds = (preds.squeeze(1) > thresholds[i]).int()
        all_preds[i, ...] = binary_preds
        all_labels[i, ...] = binary_labels
    tp = (all_preds * all_labels).sum(dim=(3, 4))
    fg_labels = all_labels.sum(dim=(3, 4))
    fg_preds = all_preds.sum(dim=(3, 4))

    if dataset == 'egteagaze':
        fixation_idx = 1
    elif dataset in ['ego4dgaze', 'ego4dgaze_forecast', 'ego4d_av_gaze', 'ego4d_av_gaze_forecast', 'aria_gaze',
                     'aria_gaze_forecast', 'aria_av_gaze', 'aria_av_gaze_forecast']:
        fixation_idx = 0
    else:
        raise NotImplementedError(f'Metrics of {dataset} is not implemented.')
    labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
    tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
    tp = tp.view(tp.size(0), tp.size(1)*tp.size(2)).index_select(1, tracked_idx)
    fg_labels = fg_labels.view(fg_labels.size(0), fg_labels.size(1)*fg_labels.size(2)).index_select(1, tracked_idx)
    fg_preds = fg_preds.view(fg_preds.size(0), fg_preds.size(1)*fg_preds.size(2)).index_select(1, tracked_idx)
    recall = (tp / (fg_labels + 1e-6)).mean(dim=1)
    precision = (tp / (fg_preds + 1e-6)).mean(dim=1)
    f1 = (2 * recall * precision) / (recall + precision + 1e-6)
    max_idx = torch.argmax(f1)

    return float(f1[max_idx].cpu().numpy()), float(recall[max_idx].cpu().numpy()), \
           float(precision[max_idx].cpu().numpy()), thresholds[max_idx]  # need np.float64 in logging rather than np.float32


def adaptive_angular_f1(preds, labels_hm, dataset):
    """
    PRG Added.
    Automatically select the angular threshold getting the best f1 score for head orientation.
    
    Args:
        preds: (B, T, 2) - predicted angular coordinates in radians
        labels_hm: (B, T, 2) - target angular coordinates in radians  
        dataset: dataset name for threshold selection
        
    Returns:
        f1, recall, precision, threshold (in degrees)
    """
    # Set appropriate angular thresholds based on dataset precision
    if 'ego4d' in dataset.lower():
        thresholds_deg = np.linspace(0.05, 0.8, 30)    
    elif 'aria' in dataset.lower(): 
        thresholds_deg = np.linspace(0.02, 0.6, 30)    
    else:  # general head orientation
        thresholds_deg = np.linspace(0.01, 1.0, 50)
    
    # Convert thresholds to radians for comparison
    thresholds_rad = np.deg2rad(thresholds_deg)
    
    # Calculate angular distances between predictions and targets
    angular_errors = torch.norm(preds - labels_hm, dim=-1)  # (B, T)
    
    # For each threshold, calculate precision/recall/F1
    best_f1 = 0.0
    best_recall = 0.0 
    best_precision = 0.0
    best_threshold = 0.0
    
    for i, threshold_rad in enumerate(thresholds_rad):
        # Binary classification: is prediction within threshold?
        correct_predictions = (angular_errors <= threshold_rad).float()  # (B, T)
        
        # Calculate metrics
        total_predictions = correct_predictions.numel()
        true_positives = correct_predictions.sum().item()
        
        # For angular accuracy, we consider all samples as "positive class"
        # So recall = accuracy within threshold
        recall = true_positives / total_predictions if total_predictions > 0 else 0.0
        
        # Precision: among predictions within threshold, how many are actually good?
        # For head orientation, precision ≈ recall (accuracy within threshold)
        precision = recall
        
        # F1 score
        f1 = (2 * recall * precision) / (recall + precision + 1e-6) if (recall + precision) > 0 else 0.0
        
        # Update best metrics
        if f1 > best_f1:
            best_f1 = f1
            best_recall = recall  
            best_precision = precision
            best_threshold = thresholds_deg[i]  # Return in degrees
    
    return float(best_f1), float(best_recall), float(best_precision), float(best_threshold)
