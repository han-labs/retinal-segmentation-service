import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model():  
    """
    Tạo mô hình U-Net++ với backbone EfficientNet-B4
    
    Returns:
        model: U-Net++ model for retinal vessel segmentation
    """
    # TODO: Add support for different backbone models
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_use_batchnorm=True,
        decoder_attention_type=None,
    )
    return model

def dice_coefficient(y_true, y_pred):
    """
    Tính Dice coefficient cho đánh giá
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        
    Returns:
        dice: Dice coefficient score
    """
    smooth = 1e-15  # This value could be tuned for better stability
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def fov_aware_dice_coefficient(targets, predictions, fov_masks, smooth=1e-7):
    """
    Tính Dice coefficient chỉ trong vùng FOV
    
    Args:
        targets: Ground truth vessel masks
        predictions: Predicted vessel masks  
        fov_masks: Field of view masks
        smooth: Smoothing factor
        
    Returns:
        dice: FOV-aware Dice coefficient
    """
    # Flatten tensors
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    fov_flat = fov_masks.view(-1)
    
    # Create mask for valid pixels (inside FOV)
    valid_mask = fov_flat > 0.5
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=targets.device)
    
    # Apply FOV mask
    predictions_masked = predictions_flat[valid_mask]
    targets_masked = targets_flat[valid_mask]
    
    # Calculate Dice
    intersection = (predictions_masked * targets_masked).sum()
    union = predictions_masked.sum() + targets_masked.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice 