"""
Knee X-ray Classification using CNN with Clinical Priorities
==========================================================

This script implements a Convolutional Neural Network (CNN) for classifying knee X-rays 
according to the Kellgren-Lawrence grading scale, with specific emphasis on clinical priorities.

Author: [Rongbin Ye & Jiaqi Chen]
Date: May 18, 2025
Version: 1.0.0
License: MIT

Data Source
----------
The knee X-ray dataset used in this project comes from: [https://www.kaggle.com/datasets/orvile/digital-knee-x-ray-images
- Dataset Name: Digital Knee X-ray Images
- Location: [Dataset URL or Reference]
- Format: Image files (X-rays) with expert annotations
- Classes: KL-Grade 0-4 (Kellgren-Lawrence grading scale)
- Annotations: Two medical expert assessments (Expert-I and Expert-II)

Clinical Priorities
-----------------
The model is optimized for the following clinical priorities:
1. High recall for severe cases (KL4) - 40% weight
2. High precision for normal cases (KL0) - 30% weight
3. High recall for moderate cases (KL3) - 20% weight
4. High precision for moderate cases - 10% weight

Requirements
-----------
- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- torchmetrics

Project Structure
---------------
.
├── data/
│   └── Digital Knee X-ray Images/
│       ├── MedicalExpert-I/
│       └── MedicalExpert-II/
├── logs/
├── models/
└── results/

Usage
-----
1. Ensure data is in the correct directory structure
2. Run: python Data_Camp.py
3. Check logs/ directory for training progress
4. Find saved models in models/ directory
5. View results and visualizations in results/ directory

Notes
-----
- The model uses ResNet-style architecture with attention mechanisms
- Implements ordinal regression loss for better grade ordering
- Includes comprehensive evaluation metrics and visualizations
- Optimized for clinical utility with weighted priorities
"""

###### Importing the libraries ######
# Standard libraries
import os
import json
import logging
from datetime import datetime

# Data handling and numerical computations
import numpy as np
import pandas as pd

# Deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Computer vision and image processing
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics and evaluation
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torchmetrics import Precision, Recall, F1Score

# Comment out LIME imports for now
# from lime import lime_image
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from PIL import Image

###### Importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import numpy as np
# Comment out LIME imports
# from lime import lime_image
# from lime.wrappers.scikit_image import SegmentationAlgorithm
# from PIL import Image
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import pandas as pd
from torchmetrics import Precision, Recall, F1Score
import json


## Loading the data using the existing tools
from torchvision.datasets import ImageFolder
from torchvision import transforms

### Set up logging configuration
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/data_loading_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.ToTensor(),
transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Base path for the dataset
base_path = "/Users/mega_potato/Downloads/Side_Project/CNN_Toy/data/Digital Knee X-ray Images/Digital Knee X-ray Images/Knee X-ray Images"

# Paths for both expert assessments
expert1_path = os.path.join(base_path, "MedicalExpert-I/MedicalExpert-I")
expert2_path = os.path.join(base_path, "MedicalExpert-II/MedicalExpert-II")

# Load both datasets
try:
    dataset_expert1 = ImageFolder(
        expert1_path,
        transform=train_transform,
    )
    logger.info(f"Expert 1 Dataset loaded successfully")
    logger.info(f"Expert 1 classes: {dataset_expert1.classes}")
    logger.info(f"Expert 1 class to index mapping: {dataset_expert1.class_to_idx}")
    logger.info(f"Expert 1 total samples: {len(dataset_expert1)}")

    dataset_expert2 = ImageFolder(
        expert2_path,
        transform=train_transform,
    )
    logger.info(f"Expert 2 Dataset loaded successfully")
    logger.info(f"Expert 2 classes: {dataset_expert2.classes}")
    logger.info(f"Expert 2 class to index mapping: {dataset_expert2.class_to_idx}")
    logger.info(f"Expert 2 total samples: {len(dataset_expert2)}")

except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")

def print_dataset_info(dataset, expert_name):
    """Helper function to print and compare dataset information"""
    logger.info(f"\n{expert_name} Dataset Information:")
    logger.info("-------------------------")
    
    # Get sample image and label
    image, label = next(iter(dataset))
    logger.info(f"Sample image shape: {image.shape}")
    logger.info(f"Sample label: {label} (Class: {dataset.classes[label]})")
    
    # Get class distribution
    class_counts = {dataset.classes[i]: 0 for i in range(len(dataset.classes))}
    for _, label in dataset:
        class_counts[dataset.classes[label]] += 1
    
    logger.info("\nClass distribution:")
    for class_name, count in class_counts.items():
        logger.info(f"{class_name}: {count} images")

# Print information for both datasets
print_dataset_info(dataset_expert1, "Medical Expert 1")
print_dataset_info(dataset_expert2, "Medical Expert 2")

# Optional: Check if the class labels match between experts
if dataset_expert1.classes == dataset_expert2.classes:
    logger.info("\nBoth experts use the same class labels")
else:
    logger.info("\nWarning: Experts use different class labels:")
    logger.info(f"Expert 1: {dataset_expert1.classes}")
    logger.info(f"Expert 2: {dataset_expert2.classes}")

####### Testing the dataset loading #########################################################
def test_dataset_loading(dataset, expert_name):
    """
    Test function to verify dataset loading and display key information
    """
    logger.info(f"\nTesting {expert_name} dataset...")
    
    # 1. Test dataset size
    assert len(dataset) > 0, "Dataset is empty!"
    logger.info(f"✓ Dataset contains {len(dataset)} samples")
    
    # 2. Test if we can access an image and its label
    sample_img, sample_label = dataset[0]
    logger.info(f"✓ First image shape: {sample_img.shape}")
    logger.info(f"✓ First image label: {dataset.classes[sample_label]}")
    
    # 3. Verify image dimensions
    assert sample_img.shape == (3, 128, 128), f"Unexpected image shape: {sample_img.shape}"
    logger.info("✓ Image dimensions are correct (3, 128, 128)")
    
    # 4. Print class distribution
    class_counts = {dataset.classes[i]: 0 for i in range(len(dataset.classes))}
    for _, label in dataset:
        class_counts[dataset.classes[label]] += 1
    
    logger.info("\nClass distribution:")
    for class_name, count in class_counts.items():
        logger.info(f"{class_name}: {count} images")
    
    logger.info(f"\n{expert_name} dataset testing completed successfully! ✓")
    return True

# Run tests for both datasets
try:
    test_dataset_loading(dataset_expert1, "Medical Expert 1")
    test_dataset_loading(dataset_expert2, "Medical Expert 2")
except Exception as e:
    logger.error(f"Dataset testing failed: {str(e)}")

#### END OF TESTING ############################################################################

###### Creating the model #####################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.num_classes = num_classes
        # Weights for each boundary between classes
        self.weights = weights if weights is not None else torch.ones(num_classes - 1)
    
    def forward(self, predictions, targets):
        # Convert targets to ordinal encoding
        ordinal_targets = torch.zeros(targets.size(0), self.num_classes - 1)
        for i in range(targets.size(0)):
            ordinal_targets[i, :targets[i]] = 1
        
        ordinal_targets = ordinal_targets.to(predictions.device)
        
        # Calculate binary cross entropy for each ordinal level
        loss = F.binary_cross_entropy_with_logits(
            predictions, ordinal_targets, 
            reduction='none'
        )
        
        # Apply class weights
        weighted_loss = loss * self.weights.to(predictions.device)
        return weighted_loss.mean()

class ImprovedNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Initial convolution with larger kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with attention
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            AttentionBlock(64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            AttentionBlock(128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            AttentionBlock(256)
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes - 1)  # One less output for ordinal regression
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# For this example, we'll use Expert 1's dataset
# Split dataset into train and test sets
from torch.utils.data import random_split, DataLoader

# Set random seed for reproducibility
torch.manual_seed(424)

# Combine datasets from both experts
combined_dataset = combine_expert_datasets(dataset_expert1, dataset_expert2)

# Calculate split sizes for combined dataset
total_size = len(combined_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Split the combined dataset
train_dataset, test_dataset = random_split(
    combined_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(424)
)

# Create data loaders with the combined dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

logger.info(f"\nCombined Dataset Split:")
logger.info(f"Total samples: {total_size}")
logger.info(f"Training samples: {len(train_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")

# Save dataset statistics
dataset_stats = {
    'total_samples': total_size,
    'train_samples': len(train_dataset),
    'test_samples': len(test_dataset),
    'classes': combined_dataset.classes
}

with open('results/visualizations/dataset_statistics.json', 'w') as f:
    json.dump(dataset_stats, f, indent=4)

# Initialize model with clinical priorities
# Weights for boundaries between classes (4 boundaries for 5 classes)
class_weights = torch.tensor([2.0, 1.5, 1.5, 2.0])  # Adjusted for ordinal boundaries
criterion = OrdinalRegressionLoss(num_classes=5, weights=class_weights)
net = ImprovedNet(num_classes=5)
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Set device
device = torch.device('cpu')
net = net.to(device)

def evaluate_clinical_metrics(model, data_loader, device, class_names):
    """
    Evaluate model with emphasis on clinical priorities
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Initialize metrics
    precision = Precision(task="multiclass", num_classes=5, average=None).to(device)
    recall = Recall(task="multiclass", num_classes=5, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=5, average=None).to(device)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Convert ordinal outputs to class predictions
            probs = torch.sigmoid(outputs)
            preds = torch.sum(probs > 0.5, dim=1)
            preds = torch.clamp(preds, 0, 4)  # Ensure predictions are within valid range
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            precision(preds, labels)
            recall(preds, labels)
            f1(preds, labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate Cohen's Kappa with quadratic weights
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(all_preds - all_labels))
    
    # Get per-class metrics
    precision_values = precision.compute()
    recall_values = recall.compute()
    f1_values = f1.compute()
    
    # Create detailed metrics dictionary
    metrics = {
        'per_class_metrics': {
            class_name: {
                'precision': float(precision_values[i]),
                'recall': float(recall_values[i]),
                'f1': float(f1_values[i]),
                'support': int(np.sum(all_labels == i))
            } for i, class_name in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist(),
        'normalized_confusion_matrix': cm_normalized.tolist(),
        'cohen_kappa': float(kappa),
        'mae': float(mae),
        'clinical_priorities': {
            'severe_recall': float(recall_values[4]),  # KL4/Severe
            'normal_precision': float(precision_values[0]),  # KL0/Normal
            'moderate_metrics': {
                'recall': float(recall_values[3]),  # KL3/Moderate
                'precision': float(precision_values[3])
            }
        }
    }
    
    # Calculate Clinical Utility Score (weighted average of priority metrics)
    clinical_utility = (
        0.4 * metrics['clinical_priorities']['severe_recall'] +
        0.3 * metrics['clinical_priorities']['normal_precision'] +
        0.2 * metrics['clinical_priorities']['moderate_metrics']['recall'] +
        0.1 * metrics['clinical_priorities']['moderate_metrics']['precision']
    )
    metrics['clinical_utility_score'] = float(clinical_utility)
    
    return metrics

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', normalize=False):
    """
    Plot confusion matrix with emphasis on critical misclassifications
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(f'confusion_matrix{"_normalized" if normalize else ""}.png')
    plt.close()

# Add after the existing visualization functions and before the training loop

def save_epoch_visualizations(metrics, epoch, training_history):
    """
    Save all visualizations for a given epoch in a structured folder system
    
    Args:
        metrics: Dictionary containing both standard and clinical metrics
        epoch: Current epoch number
        training_history: Dictionary containing training history
    """
    # Create epoch-specific directories
    epoch_dir = f'results/visualizations/epoch_{epoch:03d}'
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 1. Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standard confusion matrix
    sns.heatmap(metrics['standard']['confusion_matrix'], annot=True, fmt='d', ax=ax1,
                xticklabels=dataset_expert1.classes, yticklabels=dataset_expert1.classes)
    ax1.set_title(f'Standard Evaluation\nConfusion Matrix (Epoch {epoch})')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Clinical confusion matrix
    sns.heatmap(metrics['clinical']['confusion_matrix'], annot=True, fmt='d', ax=ax2,
                xticklabels=dataset_expert1.classes, yticklabels=dataset_expert1.classes)
    ax2.set_title(f'Clinically Acceptable\nConfusion Matrix (Epoch {epoch})')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{epoch_dir}/confusion_matrices.png')
    plt.close()
    
    # 2. Per-class Metrics Comparison
    metrics_comparison = pd.DataFrame({
        'Standard Precision': [metrics['standard']['per_class'][f'KL{i}']['precision'] for i in range(len(dataset_expert1.classes))],
        'Clinical Precision': [metrics['clinical']['per_class'][f'KL{i}']['precision'] for i in range(len(dataset_expert1.classes))],
        'Standard Recall': [metrics['standard']['per_class'][f'KL{i}']['recall'] for i in range(len(dataset_expert1.classes))],
        'Clinical Recall': [metrics['clinical']['per_class'][f'KL{i}']['recall'] for i in range(len(dataset_expert1.classes))],
        'Standard F1': [metrics['standard']['per_class'][f'KL{i}']['f1'] for i in range(len(dataset_expert1.classes))],
        'Clinical F1': [metrics['clinical']['per_class'][f'KL{i}']['f1'] for i in range(len(dataset_expert1.classes))]
    }, index=dataset_expert1.classes)
    
    # Plot per-class metrics
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    metrics_comparison[['Standard Precision', 'Clinical Precision']].plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Precision Comparison (Epoch {epoch})')
    axes[0].set_ylabel('Precision')
    
    metrics_comparison[['Standard Recall', 'Clinical Recall']].plot(kind='bar', ax=axes[1])
    axes[1].set_title(f'Recall Comparison (Epoch {epoch})')
    axes[1].set_ylabel('Recall')
    
    metrics_comparison[['Standard F1', 'Clinical F1']].plot(kind='bar', ax=axes[2])
    axes[2].set_title(f'F1 Score Comparison (Epoch {epoch})')
    axes[2].set_ylabel('F1 Score')
    
    for ax in axes:
        ax.set_xlabel('KL Grade')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{epoch_dir}/metrics_comparison.png')
    plt.close()
    
    # 3. Training Progress
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['epoch'], training_history['standard_kappa'], label='Standard Kappa')
    plt.plot(training_history['epoch'], training_history['clinical_kappa'], label='Clinical Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('Cohen\'s Kappa')
    plt.title(f'Training Progress: Standard vs Clinical Metrics (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{epoch_dir}/training_progress.png')
    plt.close()
    
    # 4. Save metrics to JSON for this epoch
    with open(f'{epoch_dir}/metrics.json', 'w') as f:
        json.dump({
            'epoch': epoch,
            'standard_metrics': metrics['standard'],
            'clinical_metrics': metrics['clinical']
        }, f, indent=4, cls=NumpyEncoder)

# Modify the training loop to include the new visualization saving
# Replace the existing training loop with this updated version

# Training loop with both standard and clinical evaluation
num_epochs = 30
best_clinical_kappa = 0
logger.info("\nStarting training with clinical priorities...")

# Create directories for saving results
os.makedirs('results/visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize lists to track metrics
training_history = {
    'standard_kappa': [],
    'clinical_kappa': [],
    'epoch': []
}

for epoch in range(num_epochs):
    # Training phase
    net.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 50 == 49:
            logger.info(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
    
    # Evaluation phase with both metrics
    metrics = evaluate_test_metrics(net, test_loader, device)
    
    # Track metrics history
    training_history['standard_kappa'].append(metrics['standard']['kappa'])
    training_history['clinical_kappa'].append(metrics['clinical']['kappa'])
    training_history['epoch'].append(epoch + 1)
    
    # Save visualizations for this epoch
    save_epoch_visualizations(metrics, epoch + 1, training_history)
    
    # Log results
    logger.info(f'\nEpoch {epoch + 1} Evaluation:')
    logger.info(f'Standard Kappa: {metrics["standard"]["kappa"]:.3f}')
    logger.info(f'Clinical Kappa: {metrics["clinical"]["kappa"]:.3f}')
    
    # Update learning rate based on clinical kappa
    scheduler.step(metrics['clinical']['kappa'])
    
    # Save best model based on clinical kappa
    if metrics['clinical']['kappa'] > best_clinical_kappa:
        best_clinical_kappa = metrics['clinical']['kappa']
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'clinical_kappa': best_clinical_kappa,
            'training_history': training_history
        }, 'models/best_clinical_model.pth')
        
        # Create a symbolic link to the best epoch's visualizations
        best_epoch_link = 'results/visualizations/best_epoch'
        if os.path.exists(best_epoch_link):
            os.remove(best_epoch_link)
        os.symlink(f'epoch_{epoch+1:03d}', best_epoch_link)
        
        logger.info(f'\nNew best model saved with clinical kappa: {best_clinical_kappa:.3f}')

logger.info("\nTraining completed!")
logger.info(f"Best clinical kappa: {best_clinical_kappa:.3f}")

# Save final training history
training_history_df = pd.DataFrame({
    'Epoch': training_history['epoch'],
    'Standard Kappa': training_history['standard_kappa'],
    'Clinical Kappa': training_history['clinical_kappa']
})
training_history_df.to_csv('results/visualizations/training_history.csv', index=False)

# Create an HTML report summarizing the training
with open('results/visualizations/training_report.html', 'w') as f:
    f.write(f"""
    <html>
    <head>
        <title>Training Report - Knee X-ray Classification</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .epoch-link {{ margin: 10px 0; }}
            .metrics {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Training Report - Knee X-ray Classification</h1>
        <h2>Best Model Performance</h2>
        <div class="metrics">
            <p>Best Clinical Kappa: {best_clinical_kappa:.3f}</p>
            <p>Achieved at Epoch: {training_history['clinical_kappa'].index(max(training_history['clinical_kappa'])) + 1}</p>
        </div>
        <h2>Epoch-wise Results</h2>
        <div class="epochs">
            {chr(10).join(f'<div class="epoch-link"><a href="epoch_{epoch:03d}/metrics.json">Epoch {epoch} Metrics</a></div>' for epoch in range(1, num_epochs + 1))}
        </div>
    </body>
    </html>
    """)

#### END OF TRAINING AND EVALUATION ############################################################################

###### Clinical Acceptable Evaluation Metrics #####################################################################
def calculate_clinical_acceptable_metrics(all_preds, all_labels, num_classes=5):
    """
    Calculate metrics with clinical acceptance criteria:
    - Up one level prediction is acceptable for KL1-KL4
    - For KL0 (normal), only correct prediction is acceptable
    - Doubtful (KL1) predictions maintain original criteria
    
    Args:
        all_preds: numpy array of model predictions
        all_labels: numpy array of true labels
        num_classes: number of KL grades (default: 5)
    
    Returns:
        Dictionary containing both standard and clinically acceptable metrics
    """
    # Convert to numpy arrays if they're not already
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate standard metrics
    standard_cm = confusion_matrix(all_labels, all_preds)
    standard_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Initialize clinically acceptable confusion matrix
    clinical_cm = np.zeros_like(standard_cm)
    
    # Fill clinical confusion matrix with adjusted criteria
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            count = np.sum((all_labels == true_label) & (all_preds == pred_label))
            
            # Apply clinical acceptance criteria
            if true_label == 0:  # Normal (KL0)
                # Only exact matches are acceptable
                clinical_cm[true_label, pred_label] = count
            elif true_label == 1:  # Doubtful (KL1)
                # Keep original criteria
                clinical_cm[true_label, pred_label] = count
            else:  # KL2-KL4
                # Accept prediction if it's the same or one level up
                if pred_label == true_label or pred_label == true_label + 1:
                    clinical_cm[true_label, true_label] += count  # Count as correct prediction
                else:
                    clinical_cm[true_label, pred_label] = count
    
    # Calculate metrics for both standard and clinical evaluation
    metrics = {
        'standard': {
            'confusion_matrix': standard_cm,
            'kappa': standard_kappa,
            'per_class': calculate_per_class_metrics(standard_cm)
        },
        'clinical': {
            'confusion_matrix': clinical_cm,
            'kappa': cohen_kappa_score(all_labels, all_preds, weights='quadratic'),
            'per_class': calculate_per_class_metrics(clinical_cm)
        }
    }
    
    return metrics

def calculate_per_class_metrics(confusion_matrix):
    """
    Calculate precision, recall, and F1 score for each class from confusion matrix
    """
    n_classes = confusion_matrix.shape[0]
    metrics = {}
    
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'KL{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def visualize_comparison_metrics(standard_metrics, clinical_metrics, class_names):
    """
    Create visualizations comparing standard and clinical metrics
    """
    # Create directory for comparison visualizations
    if not os.path.exists('results/comparison'):
        os.makedirs('results/comparison')
    
    # Plot confusion matrices side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standard confusion matrix
    sns.heatmap(standard_metrics['confusion_matrix'], annot=True, fmt='d', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Standard Evaluation\nConfusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Clinical confusion matrix
    sns.heatmap(clinical_metrics['confusion_matrix'], annot=True, fmt='d', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Clinically Acceptable\nConfusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('results/comparison/confusion_matrices_comparison.png')
    plt.close()
    
    # Compare per-class metrics
    metrics_comparison = pd.DataFrame({
        'Standard Precision': [standard_metrics['per_class'][f'KL{i}']['precision'] for i in range(len(class_names))],
        'Clinical Precision': [clinical_metrics['per_class'][f'KL{i}']['precision'] for i in range(len(class_names))],
        'Standard Recall': [standard_metrics['per_class'][f'KL{i}']['recall'] for i in range(len(class_names))],
        'Clinical Recall': [clinical_metrics['per_class'][f'KL{i}']['recall'] for i in range(len(class_names))],
        'Standard F1': [standard_metrics['per_class'][f'KL{i}']['f1'] for i in range(len(class_names))],
        'Clinical F1': [clinical_metrics['per_class'][f'KL{i}']['f1'] for i in range(len(class_names))]
    }, index=class_names)
    
    # Plot per-class metrics comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    metrics_comparison[['Standard Precision', 'Clinical Precision']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Precision Comparison')
    axes[0].set_ylabel('Precision')
    
    metrics_comparison[['Standard Recall', 'Clinical Recall']].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Recall Comparison')
    axes[1].set_ylabel('Recall')
    
    metrics_comparison[['Standard F1', 'Clinical F1']].plot(kind='bar', ax=axes[2])
    axes[2].set_title('F1 Score Comparison')
    axes[2].set_ylabel('F1 Score')
    
    for ax in axes:
        ax.set_xlabel('KL Grade')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison/metrics_comparison.png')
    plt.close()
    
    return metrics_comparison

# Modify the evaluate_test_metrics function to include clinical acceptable metrics
def evaluate_test_metrics(net, test_loader, device, num_classes=5):
    """
    Evaluate model performance on test set using both standard and clinically acceptable metrics
    """
    net.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate both standard and clinical metrics
    metrics = calculate_clinical_acceptable_metrics(all_preds, all_labels, num_classes)
    
    # Visualize comparisons
    metrics_comparison = visualize_comparison_metrics(
        metrics['standard'],
        metrics['clinical'],
        dataset_expert1.classes
    )
    
    # Log results
    logger.info("\nTest Set Evaluation Results:")
    logger.info("\nStandard Metrics:")
    logger.info(f"Cohen's Kappa: {metrics['standard']['kappa']:.3f}")
    
    logger.info("\nClinically Acceptable Metrics:")
    logger.info(f"Cohen's Kappa: {metrics['clinical']['kappa']:.3f}")
    
    logger.info("\nMetrics Comparison:")
    logger.info("\n" + str(metrics_comparison))
    
    # Save metrics to JSON
    with open('results/comparison/evaluation_metrics.json', 'w') as f:
        json.dump({
            'standard_metrics': metrics['standard'],
            'clinical_metrics': metrics['clinical']
        }, f, indent=4, cls=NumpyEncoder)
    
    return metrics

# Add NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# Run evaluation on test set
logger.info("\nEvaluating model metrics on test set...")
evaluate_test_metrics(net, test_loader, device)

#### END OF MODEL EVALUATION METRICS ##################################################################
######### END OF THE SCRIPT Version 1.0.0 ##########################################################
logger.info("\nScript execution completed. Thank you for using the Knee X-ray Classification System.")

# After the data loading section and before the training loop

def combine_expert_datasets(dataset1, dataset2):
    """
    Combine datasets from both experts, handling potential discrepancies
    
    Args:
        dataset1: ImageFolder dataset from Expert 1
        dataset2: ImageFolder dataset from Expert 2
    
    Returns:
        Combined dataset with consensus labels
    """
    logger.info("\nCombining datasets from both experts...")
    
    # Get file paths and labels from both datasets
    files1 = [sample[0] for sample in dataset1.samples]
    labels1 = [sample[1] for sample in dataset1.samples]
    files2 = [sample[0] for sample in dataset2.samples]
    labels2 = [sample[1] for sample in dataset2.samples]
    
    # Create mapping between file paths
    common_files = {}
    for f1, l1 in zip(files1, labels1):
        img_name = os.path.basename(f1)
        common_files[img_name] = {'expert1': l1, 'path': f1}
    
    for f2, l2 in zip(files2, labels2):
        img_name = os.path.basename(f2)
        if img_name in common_files:
            common_files[img_name]['expert2'] = l2
    
    # Create consensus dataset
    consensus_samples = []
    agreement_stats = {
        'full_agreement': 0,
        'one_level_diff': 0,
        'major_diff': 0
    }
    
    for img_name, data in common_files.items():
        if 'expert2' in data:  # Image exists in both datasets
            label1 = data['expert1']
            label2 = data['expert2']
            
            # Check agreement level
            if label1 == label2:
                consensus_label = label1
                agreement_stats['full_agreement'] += 1
            elif abs(label1 - label2) == 1:
                # For one level difference, use the higher grade
                consensus_label = max(label1, label2)
                agreement_stats['one_level_diff'] += 1
            else:
                # For major differences, skip the sample
                agreement_stats['major_diff'] += 1
                continue
            
            consensus_samples.append((data['path'], consensus_label))
    
    # Log agreement statistics
    total_samples = len(consensus_samples)
    logger.info("\nExpert Agreement Statistics:")
    logger.info(f"Total samples in consensus dataset: {total_samples}")
    logger.info(f"Full agreement: {agreement_stats['full_agreement']} ({100*agreement_stats['full_agreement']/total_samples:.1f}%)")
    logger.info(f"One level difference: {agreement_stats['one_level_diff']} ({100*agreement_stats['one_level_diff']/total_samples:.1f}%)")
    logger.info(f"Major differences (excluded): {agreement_stats['major_diff']}")
    
    # Create a custom dataset class for the combined data
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform
            self.classes = dataset1.classes
            self.class_to_idx = dataset1.class_to_idx
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = torchvision.io.read_image(path)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Create the combined dataset
    combined_dataset = CombinedDataset(consensus_samples, 
                                     transform=train_transform)
    
    # Visualize label distribution
    label_dist = [0] * len(dataset1.classes)
    for _, label in consensus_samples:
        label_dist[label] += 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(dataset1.classes, label_dist)
    plt.title('Label Distribution in Combined Dataset')
    plt.xlabel('KL Grade')
    plt.ylabel('Number of Samples')
    plt.savefig('results/visualizations/combined_dataset_distribution.png')
    plt.close()
    
    return combined_dataset

# Combine datasets from both experts
combined_dataset = combine_expert_datasets(dataset_expert1, dataset_expert2)

# Calculate split sizes for combined dataset
total_size = len(combined_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Split the combined dataset
train_dataset, test_dataset = random_split(
    combined_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(424)
)

# Create data loaders with the combined dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

logger.info(f"\nCombined Dataset Split:")
logger.info(f"Total samples: {total_size}")
logger.info(f"Training samples: {len(train_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")

# Save dataset statistics
dataset_stats = {
    'total_samples': total_size,
    'train_samples': len(train_dataset),
    'test_samples': len(test_dataset),
    'class_distribution': label_dist,
    'classes': combined_dataset.classes
}

with open('results/visualizations/dataset_statistics.json', 'w') as f:
    json.dump(dataset_stats, f, indent=4)


