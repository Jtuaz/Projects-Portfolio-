import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import nibabel as NIBABEL   # Used for loading and manipulating NIfTI (.nii/.nii.gz) neuroimaging files
from scipy.ndimage import zoom as ZOOM_IMAGE   # Used for 3D array interpolation/resampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as PANDAS   # Used for handling CSV metadata via DataFrames
import matplotlib.pyplot as PLOT

import torch
import torch.nn as NN
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm as PROGRESS_BAR   # Terminal progress bar for long training loops
from torchvision.models.video import r3d_18

# Global Configurations & Hyperparameters
DATASET_ROOT = r"D:\AI_Project_Dataset\OASIS-1_Process"   # Root folder containing MRI subdirectories
CSV_FILE_PATH = os.path.join(DATASET_ROOT, "oasis_labels.csv")   # Path to CSV containing metadata and labels
DATA_FRAME = PANDAS.read_csv(CSV_FILE_PATH)   # Load CSV metadata into a pandas DataFrame

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # Automatically use NVIDIA GPU if available
NUMBER_OF_CLASSES = 3   # Classification targets (e.g., Cognitive Status/Dementia Severity)
TARGET_SHAPE = (128, 128, 128)   # Fixed (Depth, Height, Width) resolution to standardize MRI inputs
BATCH_SIZE = 8   # Number of 3D volumes processed per GPU forward/backward pass
NUMBER_OF_EPOCHS = 5   # Total training passes through the dataset
LEARNING_RATE = 1e-4   # Step size multiplier for gradient descent updates
WEIGHT_DECAY = 1e-5   # L2 regularization parameter to prevent overfitting
PIN_MEMORY = True   # Speeds up PyTorch data transfer from CPU RAM to GPU RAM
NUMBER_OF_WORKERS = 12   # Multi-threading data loading CPU subprocesses
SAVE_DIRECTORY = "checkpoints_oasis"   # Local folder to store trained model weights/checkpoints
os.makedirs(SAVE_DIRECTORY, exist_ok=True)   # Ensure save directory exists (won't crash if already present)
RANDOM_SEED = 42   # Seed integer for deterministic/reproducible results

# --- Randomness Seeding for Reproducibility ---
random.seed(RANDOM_SEED)   # Sets python built-in random module seed
np.random.seed(RANDOM_SEED)   # Sets NumPy pseudo-random generator seed
torch.manual_seed(RANDOM_SEED)   # Sets PyTorch CPU random seed
if COMPUTE_DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)   # Sets PyTorch GPU random seed across all graphics cards


# Data Preprocessing & Helper Functions

def find_scan_file(DATASET_ROOT: str, SESSION_IDENTIFIER: str, SCAN_IDENTIFIER: str) -> str:
    """
    Locates the matching brain-extracted (BET) NIfTI file path
    using session IDs and scan identifiers.
    """
    BRAIN_EXTRACTION_DIRECTORY = os.path.join(DATASET_ROOT, SESSION_IDENTIFIER,
                                              "BET")   # Target Brain Extraction Tool directory

     # Check if the folder exists; if not, search for wildcard directory variants
    if not os.path.isdir(BRAIN_EXTRACTION_DIRECTORY):
        SESSION_GLOB_PATTERN = glob.glob(os.path.join(DATASET_ROOT, f"{SESSION_IDENTIFIER}*"))
        for SESSION_PATH in SESSION_GLOB_PATTERN:
            ALTERNATIVE_DIRECTORY = os.path.join(SESSION_PATH, "BET")
            if os.path.isdir(ALTERNATIVE_DIRECTORY):
                BRAIN_EXTRACTION_DIRECTORY = ALTERNATIVE_DIRECTORY
                break

     # Search for scan files inside the BET folder matching SCAN_IDENTIFIER extension patterns
    FILE_PATTERN = os.path.join(BRAIN_EXTRACTION_DIRECTORY, f"*{SCAN_IDENTIFIER}*")
    MATCHED_FILES = glob.glob(FILE_PATTERN + ".nii") + glob.glob(FILE_PATTERN + ".nii.gz") + glob.glob(
        FILE_PATTERN + "*")

     # Fallback to taking any available file inside BET if specific match fails
    if not MATCHED_FILES:
        MATCHED_FILES = glob.glob(os.path.join(DATASET_ROOT, SESSION_IDENTIFIER, "BET", "*"))

     # Return file matching SCAN_IDENTIFIER string explicitly
    for MATCHED_FILE in MATCHED_FILES:
        if SCAN_IDENTIFIER in os.path.basename(MATCHED_FILE):
            return MATCHED_FILE

     # Return first available match or None if completely empty
    return MATCHED_FILES[0] if MATCHED_FILES else None


def load_nifti(FILE_PATH: str) -> np.ndarray:
    """
    Loads a .nii or .nii.gz file into a 3D float32 NumPy array.
    """
    NIFTI_IMAGE = NIBABEL.load(FILE_PATH)   # Open file stream using Nibabel
    return NIFTI_IMAGE.get_fdata(dtype=np.float32)   # Extract raw 3D array data cast to 32-bit float


def resample_to_shape(VOLUME_ARRAY: np.ndarray, TARGET_SHAPE: Tuple[int, int, int]) -> np.ndarray:
    """
    Resizes a 3D medical volume to fixed dimensions (128, 128, 128)
    using trilinear interpolation.
    """
     # Calculate scale factor ratio along each spatial axis (X, Y, Z)
    ZOOM_FACTORS = [TARGET_DIMENSION / SOURCE_DIMENSION for SOURCE_DIMENSION, TARGET_DIMENSION in
                    zip(VOLUME_ARRAY.shape, TARGET_SHAPE)]

     # Apply zoom interpolation (order=1 means Linear/Trilinear interpolation)
    return ZOOM_IMAGE(VOLUME_ARRAY, zoom=ZOOM_FACTORS, order=1)


def normalize_intensity(VOLUME_ARRAY: np.ndarray) -> np.ndarray:
    """
    Standardizes image intensities: Clips extreme outliers (1st to 99th percentile)
    and applies Z-score normalization (mean 0, std 1).
    """
    WORKING_VOLUME = VOLUME_ARRAY.copy()

     # Intensity Clipping: removes noise/outlier artifacts at extreme top & bottom percentiles
    FIRST_PERCENTILE, NINETY_NINTH_PERCENTILE = np.percentile(WORKING_VOLUME, (1, 99))
    WORKING_VOLUME = np.clip(WORKING_VOLUME, FIRST_PERCENTILE, NINETY_NINTH_PERCENTILE)

     # Z-Score Standardisation: (voxel_value - mean) / standard_deviation
    VOLUME_MEAN = WORKING_VOLUME.mean()
    VOLUME_STANDARD_DEVIATION = WORKING_VOLUME.std() if WORKING_VOLUME.std() > 0 else 1.0   # Guard against division by zero

    WORKING_VOLUME = (WORKING_VOLUME - VOLUME_MEAN) / VOLUME_STANDARD_DEVIATION

    return WORKING_VOLUME.astype(np.float32)   # Return normalized volume array


# PyTorch Dataset Definition

class OASISDataset(Dataset):
    def __init__(self, DATA_RECORDS: List[Dict], DATASET_ROOT: str, TARGET_SHAPE: Tuple[int, int, int],
                 LABEL_MAP: Dict[str, int], APPLY_AUGMENTATION: bool = False):
        self.DATA_RECORDS = DATA_RECORDS   # List of record dictionaries containing scan metadata
        self.DATASET_ROOT = DATASET_ROOT   # Path to root dataset directory
        self.TARGET_SHAPE = TARGET_SHAPE   # Target (Depth, Height, Width) shape for resampling
        self.LABEL_MAP = LABEL_MAP   # Mapping dictionary from string labels to class integers
        self.APPLY_AUGMENTATION = APPLY_AUGMENTATION   # Flag enabling random data augmentation transformations

    def __len__(self):
        return len(self.DATA_RECORDS)   # Returns total number of samples in dataset

    def __getitem__(self, INDEX: int):
        RECORD = self.DATA_RECORDS[INDEX]   # Extract single metadata dictionary
        SESSION_IDENTIFIER = RECORD["session_id"]
        SCAN_IDENTIFIER = RECORD["scan_id"]
        LABEL_STRING = RECORD["label"]
        LABEL_INDEX = self.LABEL_MAP[LABEL_STRING]   # Convert string label to mapped integer class

        FILE_PATH = find_scan_file(self.DATASET_ROOT, SESSION_IDENTIFIER, SCAN_IDENTIFIER)
        if FILE_PATH is None:
            raise FileNotFoundError(f"No scan file found for {SESSION_IDENTIFIER} {SCAN_IDENTIFIER}")

         # Load, resample, and normalize 3D MRI scan
        VOLUME_ARRAY = load_nifti(FILE_PATH)
        VOLUME_ARRAY = resample_to_shape(VOLUME_ARRAY, self.TARGET_SHAPE)
        VOLUME_ARRAY = normalize_intensity(VOLUME_ARRAY)

         # Apply random spatial axis flipping for data augmentation
        if self.APPLY_AUGMENTATION:
            if random.random() < 0.5:
                VOLUME_ARRAY = np.flip(VOLUME_ARRAY, axis=0).copy()   # Flip along X-axis
            if random.random() < 0.5:
                VOLUME_ARRAY = np.flip(VOLUME_ARRAY, axis=1).copy()   # Flip along Y-axis

        VOLUME_ARRAY = np.expand_dims(VOLUME_ARRAY, axis=0)   # Add channel dimension: shape becomes (1, D, H, W)
        return torch.from_numpy(VOLUME_ARRAY).float(), torch.tensor(LABEL_INDEX, dtype=torch.long)


# Neural Network Model Components

class ConvBlock(NN.Module):
    def __init__(self, INPUT_CHANNELS: int, OUTPUT_CHANNELS: int):
        super().__init__()
         # First 3D convolution sequence
        self.CONVOLUTION_ONE = NN.Conv3d(INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.BATCH_NORM_ONE = NN.BatchNorm3d(OUTPUT_CHANNELS)

         # Second 3D convolution sequence
        self.CONVOLUTION_TWO = NN.Conv3d(OUTPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.BATCH_NORM_TWO = NN.BatchNorm3d(OUTPUT_CHANNELS)

        self.RELU_ACTIVATION = NN.ReLU(inplace=True)

    def forward(self, INPUT_TENSOR: torch.Tensor) -> torch.Tensor:
        INPUT_TENSOR = self.RELU_ACTIVATION(self.BATCH_NORM_ONE(self.CONVOLUTION_ONE(INPUT_TENSOR)))
        INPUT_TENSOR = self.RELU_ACTIVATION(self.BATCH_NORM_TWO(self.CONVOLUTION_TWO(INPUT_TENSOR)))
        return INPUT_TENSOR


class Down(NN.Module):
    def __init__(self, INPUT_CHANNELS: int, OUTPUT_CHANNELS: int):
        super().__init__()
        self.MAX_POOLING = NN.MaxPool3d(kernel_size=2)   # Spatial resolution downsampling by factor of 2
        self.CONVOLUTION_BLOCK = ConvBlock(INPUT_CHANNELS, OUTPUT_CHANNELS)

    def forward(self, INPUT_TENSOR: torch.Tensor) -> torch.Tensor:
        INPUT_TENSOR = self.MAX_POOLING(INPUT_TENSOR)
        INPUT_TENSOR = self.CONVOLUTION_BLOCK(INPUT_TENSOR)
        return INPUT_TENSOR


class Simple3DCnn(NN.Module):
    def __init__(self, INPUT_CHANNELS: int = 1, NUMBER_OF_CLASSES: int = 3):
        super().__init__()
        self.FEATURE_EXTRACTOR = NN.Sequential(
            NN.Conv3d(INPUT_CHANNELS, 8, kernel_size=3, padding=1),
            NN.BatchNorm3d(8),
            NN.ReLU(inplace=True),
            NN.MaxPool3d(2),

            NN.Conv3d(8, 16, kernel_size=3, padding=1),
            NN.BatchNorm3d(16),
            NN.ReLU(inplace=True),
            NN.MaxPool3d(2),

            NN.Conv3d(16, 32, kernel_size=3, padding=1),
            NN.BatchNorm3d(32),
            NN.ReLU(inplace=True),

            NN.AdaptiveAvgPool3d(1)   # Reduces spatial volume down to (1, 1, 1)
        )
        self.CLASSIFIER_HEAD = NN.Linear(32, NUMBER_OF_CLASSES)

    def forward(self, INPUT_TENSOR: torch.Tensor) -> torch.Tensor:
        INPUT_TENSOR = self.FEATURE_EXTRACTOR(INPUT_TENSOR)
        INPUT_TENSOR = INPUT_TENSOR.view(INPUT_TENSOR.size(0), -1)   # Flatten spatial dimensions into 1D vector batch
        return self.CLASSIFIER_HEAD(INPUT_TENSOR)


class VNetSimple(NN.Module):
    def __init__(self, INPUT_CHANNELS: int = 1, BASE_CHANNELS: int = 16, NUMBER_OF_CLASSES: int = 2):
        super().__init__()
         # Encoder backbone layers
        self.ENCODER_STAGE_ONE = ConvBlock(INPUT_CHANNELS, BASE_CHANNELS)
        self.DOWNSAMPLE_STAGE_ONE = Down(BASE_CHANNELS, BASE_CHANNELS * 2)
        self.DOWNSAMPLE_STAGE_TWO = Down(BASE_CHANNELS * 2, BASE_CHANNELS * 4)
        self.DOWNSAMPLE_STAGE_THREE = Down(BASE_CHANNELS * 4, BASE_CHANNELS * 8)
        self.BOTTLENECK_BLOCK = ConvBlock(BASE_CHANNELS * 8, BASE_CHANNELS * 8)

         # Classification head components
        self.GLOBAL_AVERAGE_POOLING = NN.AdaptiveAvgPool3d(1)
        self.FULLY_CONNECTED_LAYER = NN.Linear(BASE_CHANNELS * 8, NUMBER_OF_CLASSES)

    def forward(self, INPUT_TENSOR: torch.Tensor) -> torch.Tensor:
        INPUT_TENSOR = self.ENCODER_STAGE_ONE(INPUT_TENSOR)
        INPUT_TENSOR = self.DOWNSAMPLE_STAGE_ONE(INPUT_TENSOR)
        INPUT_TENSOR = self.DOWNSAMPLE_STAGE_TWO(INPUT_TENSOR)
        INPUT_TENSOR = self.DOWNSAMPLE_STAGE_THREE(INPUT_TENSOR)
        INPUT_TENSOR = self.BOTTLENECK_BLOCK(INPUT_TENSOR)

        GLOBAL_FEATURES = self.GLOBAL_AVERAGE_POOLING(INPUT_TENSOR).view(INPUT_TENSOR.size(0), -1)
        return self.FULLY_CONNECTED_LAYER(GLOBAL_FEATURES)


def buildResNet3D(NUMBER_OF_CLASSES: int):
    """
    Builds a 3D ResNet18 and adapts the input stem to accept 1-channel MRI volumes.
    """
    NEURAL_NETWORK_MODEL = r3d_18(weights=None)

     # Override original RGB 3-channel initial convolution layer to accept single-channel MRI scans
    NEURAL_NETWORK_MODEL.stem[0] = NN.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False
    )

     # Output projection to matching target classification classes
    NEURAL_NETWORK_MODEL.fc = NN.Linear(NEURAL_NETWORK_MODEL.fc.in_features, NUMBER_OF_CLASSES)
    return NEURAL_NETWORK_MODEL


# Training and Evaluation Routines

def train_one_epoch(MODEL: NN.Module, DATA_LOADER: DataLoader, OPTIMIZER: Adam, LOSS_FUNCTION: NN.Module,
                    COMPUTE_DEVICE: str, EPOCH_NUMBER: int):
    MODEL.train()   # Set model to training mode (enables dropout & batchnorm update)
    RUNNING_LOSS, CORRECT_PREDICTIONS, TOTAL_SAMPLES = 0.0, 0, 0

    PROGRESS_BAR_INSTANCE = PROGRESS_BAR(DATA_LOADER, desc=f"Epoch {EPOCH_NUMBER} [Train]", ncols=100)
    for BATCH_X, BATCH_Y in PROGRESS_BAR_INSTANCE:
        BATCH_X, BATCH_Y = BATCH_X.to(COMPUTE_DEVICE), BATCH_Y.to(COMPUTE_DEVICE)

        OPTIMIZER.zero_grad()   # Reset gradients before backward pass
        MODEL_OUTPUTS = MODEL(BATCH_X)
        LOSS_VALUE = LOSS_FUNCTION(MODEL_OUTPUTS, BATCH_Y)
        LOSS_VALUE.backward()   # Backpropagate loss gradients
        OPTIMIZER.step()   # Apply parameter weights update

        RUNNING_LOSS += LOSS_VALUE.item() * BATCH_X.size(0)
        PREDICTED_CLASSES = MODEL_OUTPUTS.argmax(dim=1)
        CORRECT_PREDICTIONS += (PREDICTED_CLASSES == BATCH_Y).sum().item()
        TOTAL_SAMPLES += BATCH_X.size(0)

        PROGRESS_BAR_INSTANCE.set_postfix({
            "Loss": f"{RUNNING_LOSS / TOTAL_SAMPLES:.4f}",
            "Acc": f"{CORRECT_PREDICTIONS / TOTAL_SAMPLES:.3f}"
        })

    return RUNNING_LOSS / TOTAL_SAMPLES, CORRECT_PREDICTIONS / TOTAL_SAMPLES


def evaluate(MODEL: NN.Module, DATA_LOADER: DataLoader, LOSS_FUNCTION: NN.Module, COMPUTE_DEVICE: str,
             EPOCH_NUMBER: int):
    MODEL.eval()   # Set model to evaluation mode (disables dropout & batchnorm updates)
    RUNNING_LOSS, CORRECT_PREDICTIONS, TOTAL_SAMPLES = 0.0, 0, 0

    PROGRESS_BAR_INSTANCE = PROGRESS_BAR(DATA_LOADER, desc=f"Epoch {EPOCH_NUMBER} [Val]", ncols=100)
    with torch.no_grad():   # Disable gradient calculation graph to save RAM and time
        for BATCH_X, BATCH_Y in PROGRESS_BAR_INSTANCE:
            BATCH_X, BATCH_Y = BATCH_X.to(COMPUTE_DEVICE), BATCH_Y.to(COMPUTE_DEVICE)
            MODEL_OUTPUTS = MODEL(BATCH_X)
            LOSS_VALUE = LOSS_FUNCTION(MODEL_OUTPUTS, BATCH_Y)

            RUNNING_LOSS += LOSS_VALUE.item() * BATCH_X.size(0)
            PREDICTED_CLASSES = MODEL_OUTPUTS.argmax(dim=1)
            CORRECT_PREDICTIONS += (PREDICTED_CLASSES == BATCH_Y).sum().item()
            TOTAL_SAMPLES += BATCH_X.size(0)

            PROGRESS_BAR_INSTANCE.set_postfix({
                "Loss": f"{RUNNING_LOSS / TOTAL_SAMPLES:.4f}",
                "Acc": f"{CORRECT_PREDICTIONS / TOTAL_SAMPLES:.3f}"
            })

    return RUNNING_LOSS / TOTAL_SAMPLES, CORRECT_PREDICTIONS / TOTAL_SAMPLES


def get_predictions(MODEL: NN.Module, DATA_LOADER: DataLoader, COMPUTE_DEVICE: str):
    MODEL.eval()
    ALL_PREDICTIONS, ALL_GROUND_TRUTH_LABELS = [], []

    with torch.no_grad():
        for BATCH_X, BATCH_Y in PROGRESS_BAR(DATA_LOADER, desc="Collecting predictions"):
            BATCH_X = BATCH_X.to(COMPUTE_DEVICE)
            MODEL_OUTPUTS = MODEL(BATCH_X)
            PREDICTED_CLASSES = MODEL_OUTPUTS.argmax(dim=1).cpu().numpy()

            ALL_PREDICTIONS.extend(PREDICTED_CLASSES)
            ALL_GROUND_TRUTH_LABELS.extend(BATCH_Y.numpy())

    return np.array(ALL_GROUND_TRUTH_LABELS), np.array(ALL_PREDICTIONS)


# Execution
def main():
     # Load CSV metadata file
    DATA_FRAME = PANDAS.read_csv(CSV_FILE_PATH)
    print("CSV loaded:", len(DATA_FRAME), "rows")

    if not {"session_id", "scan_id", "label"}.issubset(DATA_FRAME.columns):
        raise ValueError("CSV must contain columns: session_id, scan_id, label")

     # Convert DataFrame rows to records list
    SCAN_RECORDS = DATA_FRAME.to_dict(orient="records")

     # Filter and keep only scans that exist on disk
    EXISTING_RECORDS = [RECORD for RECORD in SCAN_RECORDS if
                        find_scan_file(DATASET_ROOT, RECORD["session_id"], RECORD["scan_id"])]
    print("Found scans:", len(EXISTING_RECORDS))
    SCAN_RECORDS = EXISTING_RECORDS

     # Get unique patient IDs and their corresponding labels
    PATIENT_DATA_FRAME = PANDAS.DataFrame(SCAN_RECORDS).drop_duplicates(subset='session_id')
    PATIENT_IDENTIFIERS = PATIENT_DATA_FRAME['session_id'].to_numpy()
    PATIENT_LABELS = PATIENT_DATA_FRAME['label'].to_numpy()

     # Perform stratified split on patient level (ensures no data leak between train and validation)
    TRAIN_PATIENTS, VALIDATION_PATIENTS = train_test_split(
        PATIENT_IDENTIFIERS,
        test_size=0.2,
        stratify=PATIENT_LABELS,
        random_state=RANDOM_SEED
    )

     # Assign all scans belonging to train patients to training set and vice versa
    TRAINING_RECORDS = [RECORD for RECORD in SCAN_RECORDS if RECORD['session_id'] in TRAIN_PATIENTS]
    VALIDATION_RECORDS = [RECORD for RECORD in SCAN_RECORDS if RECORD['session_id'] in VALIDATION_PATIENTS]

    print("Train / Val sizes:", len(TRAINING_RECORDS), "/", len(VALIDATION_RECORDS))

     # Map target string labels to class index integers
    UNIQUE_LABELS = sorted(DATA_FRAME["label"].dropna().unique().tolist())
    LABEL_MAP = {LABEL_NAME: INDEX for INDEX, LABEL_NAME in enumerate(UNIQUE_LABELS)}
    global NUMBER_OF_CLASSES
    NUMBER_OF_CLASSES = len(UNIQUE_LABELS)
    print("Detected labels:", LABEL_MAP)

     # Generate dataset summary stats
    SCAN_COUNTS = DATA_FRAME['label'].value_counts()
    UNIQUE_PATIENTS_DATA_FRAME = DATA_FRAME.drop_duplicates(subset=['session_id'])
    PATIENT_COUNTS = UNIQUE_PATIENTS_DATA_FRAME['label'].value_counts()

    SUMMARY_TABLE = PANDAS.DataFrame({
        'Unique_Patients': PATIENT_COUNTS,
        'Total_Scans': SCAN_COUNTS
    }).fillna(0).astype(int)

    TOTAL_UNIQUE_PATIENTS = UNIQUE_PATIENTS_DATA_FRAME['session_id'].nunique()
    TOTAL_SCANS_COUNT = len(DATA_FRAME)

    print("\nDataset Summary: Patients and Scans per Diagnosis\n")
    print(SUMMARY_TABLE)
    print("\nDetailed counts per diagnosis:")
    for LABEL_NAME, ROW_DATA in SUMMARY_TABLE.iterrows():
        print(f"  {LABEL_NAME}: {ROW_DATA['Unique_Patients']} patients, {ROW_DATA['Total_Scans']} scans")
    print("\nOverall Totals")
    print(f"Total unique patients: {TOTAL_UNIQUE_PATIENTS}")
    print(f"Total scans: {TOTAL_SCANS_COUNT}")

     # Initialize PyTorch datasets and data loaders
    TRAINING_DATASET = OASISDataset(TRAINING_RECORDS, DATASET_ROOT, TARGET_SHAPE, LABEL_MAP, APPLY_AUGMENTATION=True)
    VALIDATION_DATASET = OASISDataset(VALIDATION_RECORDS, DATASET_ROOT, TARGET_SHAPE, LABEL_MAP,
                                      APPLY_AUGMENTATION=False)

    TRAIN_DATA_LOADER = DataLoader(
        TRAINING_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=PIN_MEMORY
    )

    VALIDATION_DATA_LOADER = DataLoader(
        VALIDATION_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=PIN_MEMORY
    )

     # Initialize model, loss criterion, and optimizer
    MODEL_INSTANCE = VNetSimple(INPUT_CHANNELS=1, BASE_CHANNELS=16, NUMBER_OF_CLASSES=NUMBER_OF_CLASSES).to(
        COMPUTE_DEVICE)
    LOSS_CRITERION = NN.CrossEntropyLoss()
    OPTIMIZER_INSTANCE = Adam(MODEL_INSTANCE.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

     # Training and validation loop
    BEST_VALIDATION_ACCURACY = 0.0
    for EPOCH_INDEX in range(1, NUMBER_OF_EPOCHS + 1):
        TRAIN_LOSS, TRAIN_ACCURACY = train_one_epoch(MODEL_INSTANCE, TRAIN_DATA_LOADER, OPTIMIZER_INSTANCE,
                                                     LOSS_CRITERION, COMPUTE_DEVICE, EPOCH_INDEX)
        VALIDATION_LOSS, VALIDATION_ACCURACY = evaluate(MODEL_INSTANCE, VALIDATION_DATA_LOADER, LOSS_CRITERION,
                                                        COMPUTE_DEVICE, EPOCH_INDEX)

        print(f"\nEpoch {EPOCH_INDEX:03d} Summary | Training: loss={TRAIN_LOSS:.4f}, Accuracy={TRAIN_ACCURACY:.4f} | "
              f"Validation: loss={VALIDATION_LOSS:.4f}, Accuracy={VALIDATION_ACCURACY:.4f}")

         # Checkpoint saving on accuracy improvement
        if VALIDATION_ACCURACY > BEST_VALIDATION_ACCURACY:
            BEST_VALIDATION_ACCURACY = VALIDATION_ACCURACY
            CHECKPOINT_SAVE_PATH = os.path.join(SAVE_DIRECTORY,
                                                f"vnet_best_epoch{EPOCH_INDEX}_acc{VALIDATION_ACCURACY:.4f}.pth")
            torch.save({
                "epoch": EPOCH_INDEX,
                "model_state": MODEL_INSTANCE.state_dict(),
                "optimizer_state": OPTIMIZER_INSTANCE.state_dict(),
                "val_acc": VALIDATION_ACCURACY,
                "label_map": LABEL_MAP
            }, CHECKPOINT_SAVE_PATH)
            print("Saved best checkpoint.")

     # Generate Confusion Matrix
    print("\nEvaluating confusion matrix on validation set...")
    GROUND_TRUTH_LABELS, PREDICTED_LABELS = get_predictions(MODEL_INSTANCE, VALIDATION_DATA_LOADER, COMPUTE_DEVICE)
    CONFUSION_MAT = confusion_matrix(GROUND_TRUTH_LABELS, PREDICTED_LABELS)
    CONFUSION_DISPLAY = ConfusionMatrixDisplay(confusion_matrix=CONFUSION_MAT, display_labels=list(LABEL_MAP.keys()))
    CONFUSION_DISPLAY.plot(cmap="Blues", values_format="d")
    PLOT.title("Confusion Matrix (Validation Set)")
    PLOT.show()

    print(f"\nTraining complete. Best validation accuracy: {BEST_VALIDATION_ACCURACY:.4f}")


if __name__ == "__main__":
    main()
