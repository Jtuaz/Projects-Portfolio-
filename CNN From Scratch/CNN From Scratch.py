import math
import torch
import torchvision

# TRANSFORMS IMPLEMENTED FROM SCRATCH

class CUSTOM_TO_TENSOR:
    """Converts a raw PIL Image or NumPy array (H x W x C) in [0, 255] to a FloatTensor (C x H x W) in [0.0, 1.0]."""

    def __call__(self, INPUT_IMAGE):
        # Convert integer array (0 to 255) to float tensor and scale values to floating point range [0.0, 1.0]
        TENSOR_IMAGE = torch.from_numpy(INPUT_IMAGE).float() / 255.0

        # Rearrange tensor dimensions from Height-Width-Channel (H, W, C) to PyTorch's expected Channel-Height-Width (C, H, W)
        TRANSPOSED_TENSOR = TENSOR_IMAGE.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        return TRANSPOSED_TENSOR

class CUSTOM_NORMALIZE:
    """Standardizes input tensor per channel using the formula: (X - MEAN) / STD."""

    def __init__(self, MEAN_TUPLE, STD_TUPLE):
        # Reshape 1D mean and std tuples into (3, 1, 1) tensors so broadcasting works across spatial dimensions (C, H, W)
        self.MEAN_VECTOR = torch.tensor(MEAN_TUPLE).view(3, 1, 1)
        self.STD_VECTOR = torch.tensor(STD_TUPLE).view(3, 1, 1)

    def __call__(self, INPUT_TENSOR):
        # Apply element-wise standard score formula across image channels using PyTorch broadcasting
        NORMALIZED_TENSOR = (INPUT_TENSOR - self.MEAN_VECTOR) / self.STD_VECTOR

        return NORMALIZED_TENSOR


class CUSTOM_COMPOSE_TRANSFORMS:
    """Sequentially chains multiple custom transforms together."""

    def __init__(self, TRANSFORM_LIST):
        # Store the list of transformation callables
        self.TRANSFORM_LIST = TRANSFORM_LIST

    def __call__(self, INPUT_DATA):
        PROCESSED_DATA = INPUT_DATA

        # Pass the input data sequentially through each transformation step in the list
        for INDIVIDUAL_TRANSFORM in self.TRANSFORM_LIST:
            PROCESSED_DATA = INDIVIDUAL_TRANSFORM(PROCESSED_DATA)

        return PROCESSED_DATA


# CUSTOM NEURAL NETWORK LAYERS FROM SCRATCH

class FROM_SCRATCH_CONVOLUTION_2D(torch.nn.Module):
    """2D Convolutional Layer implemented from scratch using matrix multiplication (im2col / unfold)."""

    def __init__(self, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE=3, PADDING=1, STRIDE=1):
        super(FROM_SCRATCH_CONVOLUTION_2D, self).__init__()

        self.IN_CHANNELS = IN_CHANNELS
        self.OUT_CHANNELS = OUT_CHANNELS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.PADDING = PADDING
        self.STRIDE = STRIDE

        # He / Kaiming Normal Initialization bounds to prevent vanishing/exploding gradients during training
        BOUND_VAL = 1.0 / math.sqrt(IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)

        # Initialize trainable weight tensor: shape (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)
        self.WEIGHT_TENSOR = torch.nn.Parameter(
            torch.randn(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE) * BOUND_VAL
        )

        # Initialize trainable bias vector: one bias parameter per output channel
        self.BIAS_VECTOR = torch.nn.Parameter(torch.zeros(OUT_CHANNELS))

    def forward(self, INPUT_TENSOR):
        # Extract dimensions from input batch tensor
        BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = INPUT_TENSOR.shape

        # Apply zero-padding manually to input image borders if padding > 0
        if self.PADDING > 0:
            # pad format: (pad_left, pad_right, pad_top, pad_bottom)
            PADDED_INPUT = torch.nn.functional.pad(
                INPUT_TENSOR, (self.PADDING, self.PADDING, self.PADDING, self.PADDING)
            )
        else:
            PADDED_INPUT = INPUT_TENSOR

        # Calculate spatial height and width of output feature maps
        OUTPUT_HEIGHT = (HEIGHT + 2 * self.PADDING - self.KERNEL_SIZE) // self.STRIDE + 1
        OUTPUT_WIDTH = (WIDTH + 2 * self.PADDING - self.KERNEL_SIZE) // self.STRIDE + 1

        # Extract sliding receptive field windows into flat column vectors (im2col approach)
        # Result shape: (BATCH_SIZE, IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, OUTPUT_HEIGHT * OUTPUT_WIDTH)
        UNFOLDED_COLUMNS = torch.nn.functional.unfold(
            PADDED_INPUT, kernel_size=self.KERNEL_SIZE, stride=self.STRIDE
        )

        # Flatten 4D weight kernel tensor into a 2D matrix of shape: (OUT_CHANNELS, IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
        FLATTENED_WEIGHTS = self.WEIGHT_TENSOR.view(self.OUT_CHANNELS, -1)

        # Compute convolution via matrix multiplication: (OUT_C, K_DIM) @ (BATCH, K_DIM, OUT_H * OUT_W)
        # Matrix output shape: (BATCH_SIZE, OUT_CHANNELS, OUTPUT_HEIGHT * OUTPUT_WIDTH)
        CONVOLUTION_OUTPUT = torch.matmul(FLATTENED_WEIGHTS, UNFOLDED_COLUMNS)

        # Add channel biases using broadcasting shape (1, OUT_CHANNELS, 1)
        CONVOLUTION_OUTPUT += self.BIAS_VECTOR.view(1, -1, 1)

        # Reshape flat matrix output back to 4D spatial feature map: (BATCH_SIZE, OUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        FINAL_FEATURE_MAP = CONVOLUTION_OUTPUT.view(BATCH_SIZE, self.OUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        return FINAL_FEATURE_MAP

class FROM_SCRATCH_LINEAR(torch.nn.Module):
    """Fully Connected (Dense) Layer implemented from scratch: Y = X * W^T + B."""

    def __init__(self, IN_FEATURES, OUT_FEATURES):
        super(FROM_SCRATCH_LINEAR, self).__init__()

        # Xavier / Kaiming scaling boundary based on fan-in dimension
        BOUND_VAL = 1.0 / math.sqrt(IN_FEATURES)

        # Trainable weight matrix of shape (OUT_FEATURES, IN_FEATURES)
        self.WEIGHT_MATRIX = torch.nn.Parameter(torch.randn(OUT_FEATURES, IN_FEATURES) * BOUND_VAL)

        # Trainable bias vector of shape (OUT_FEATURES)
        self.BIAS_VECTOR = torch.nn.Parameter(torch.zeros(OUT_FEATURES))

    def forward(self, INPUT_TENSOR):
        # Compute affine linear transformation: (BATCH, IN_FEATURES) @ (IN_FEATURES, OUT_FEATURES) + (OUT_FEATURES)
        DENSE_OUTPUT = torch.matmul(INPUT_TENSOR, self.WEIGHT_MATRIX.t()) + self.BIAS_VECTOR

        return DENSE_OUTPUT


class FROM_SCRATCH_RELU(torch.nn.Module):
    """Rectified Linear Unit Activation from scratch: f(x) = max(0, x)."""

    def forward(self, INPUT_TENSOR):
        # Clamp all negative input values to zero while leaving positive values untouched
        return torch.clamp(INPUT_TENSOR, min=0.0)


class FROM_SCRATCH_MAX_POOL_2D(torch.nn.Module):
    """2D Max Pooling layer implemented from scratch by reshaping spatial grids."""

    def __init__(self, KERNEL_SIZE=2, STRIDE=2):
        super(FROM_SCRATCH_MAX_POOL_2D, self).__init__()
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE

    def forward(self, INPUT_TENSOR):
        BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = INPUT_TENSOR.shape

        # Determine spatial dimensions after applying pooling stride
        OUTPUT_HEIGHT = HEIGHT // self.STRIDE
        OUTPUT_WIDTH = WIDTH // self.STRIDE

        # Reshape feature tensor to isolate pooling windows into independent tensor axes
        # Shape: (BATCH, CHANNELS, OUT_H, KERNEL_SIZE, OUT_W, KERNEL_SIZE)
        RESHAPED_TENSOR = INPUT_TENSOR.view(
            BATCH_SIZE, CHANNELS, OUTPUT_HEIGHT, self.KERNEL_SIZE, OUTPUT_WIDTH, self.KERNEL_SIZE
        )

        # Compute maximum across height window dimension (dim 3)
        MAX_POOLED_TENSOR, _ = RESHAPED_TENSOR.max(dim=3)

        # Compute maximum across width window dimension (dim 4 after reduction)
        MAX_POOLED_TENSOR, _ = MAX_POOLED_TENSOR.max(dim=4)

        return MAX_POOLED_TENSOR


class SCRATCH_CROSS_ENTROPY_LOSS(torch.nn.Module):
    """Cross-Entropy Loss function from scratch with numerically stable Softmax logic."""

    def forward(self, LOGITS_TENSOR, TARGET_LABELS):
        # Extract maximum logit value per sample in the batch for numerical stability adjustment
        MAX_LOGITS, _ = torch.max(LOGITS_TENSOR, dim=1, keepdim=True)

        # Shift logits by max value so exp(x) never overflows float ranges
        STABILIZED_LOGITS = LOGITS_TENSOR - MAX_LOGITS

        # Compute Softmax probabilities: exp(x_i) / sum(exp(x_j))
        EXPONENTIAL_LOGITS = torch.exp(STABILIZED_LOGITS)
        SOFTMAX_PROBABILITIES = EXPONENTIAL_LOGITS / torch.sum(EXPONENTIAL_LOGITS, dim=1, keepdim=True)

        # Batch indexing setup to retrieve ground truth label probabilities
        BATCH_SIZE = LOGITS_TENSOR.size(0)
        BATCH_INDICES = torch.arange(BATCH_SIZE, device=LOGITS_TENSOR.device)

        # Select predicted probabilities corresponding to the target class labels
        CORRECT_CLASS_PROBABILITIES = SOFTMAX_PROBABILITIES[BATCH_INDICES, TARGET_LABELS]

        # Calculate Negative Log Likelihood loss (add 1e-9 epsilon to prevent log(0) undefined errors)
        COMPUTED_LOSS = -torch.log(CORRECT_CLASS_PROBABILITIES + 1e-9)

        # Compute mean loss scalar across all samples in the current batch
        AVERAGE_LOSS = torch.mean(COMPUTED_LOSS)

        return AVERAGE_LOSS

# FULL CNN ARCHITECTURE ASSEMBLY

class FULL_SCRATCH_CNN(torch.nn.Module):
    def __init__(self, NUMBER_OF_CLASSES=10):
        super(FULL_SCRATCH_CNN, self).__init__()

        # Feature Extractor Block 1: Input (3, 32, 32) -> Output (16, 32, 32) -> Pooled (16, 16, 16)
        self.CONVOLUTION_LAYER_1 = FROM_SCRATCH_CONVOLUTION_2D(IN_CHANNELS=3, OUT_CHANNELS=16, KERNEL_SIZE=3, PADDING=1)
        self.RELU_ACTIVATION_1 = FROM_SCRATCH_RELU()
        self.MAX_POOLING_LAYER_1 = FROM_SCRATCH_MAX_POOL_2D(KERNEL_SIZE=2, STRIDE=2)

        # Feature Extractor Block 2: Input (16, 16, 16) -> Output (32, 16, 16) -> Pooled (32, 8, 8)
        self.CONVOLUTION_LAYER_2 = FROM_SCRATCH_CONVOLUTION_2D(IN_CHANNELS=16, OUT_CHANNELS=32, KERNEL_SIZE=3,
                                                               PADDING=1)
        self.RELU_ACTIVATION_2 = FROM_SCRATCH_RELU()
        self.MAX_POOLING_LAYER_2 = FROM_SCRATCH_MAX_POOL_2D(KERNEL_SIZE=2, STRIDE=2)

        # Classification Dense Block: Flattened Input (2048) -> Hidden (128) -> Output Logits (NUMBER_OF_CLASSES)
        self.FULLY_CONNECTED_LAYER_1 = FROM_SCRATCH_LINEAR(IN_FEATURES=32 * 8 * 8, OUT_FEATURES=128)
        self.RELU_ACTIVATION_3 = FROM_SCRATCH_RELU()
        self.FULLY_CONNECTED_LAYER_2 = FROM_SCRATCH_LINEAR(IN_FEATURES=128, OUT_FEATURES=NUMBER_OF_CLASSES)

    def forward(self, INPUT_TENSOR):
        # Pass input through Convolutional Block 1
        FEATURE_MAP_1 = self.CONVOLUTION_LAYER_1(INPUT_TENSOR)
        ACTIVATED_MAP_1 = self.RELU_ACTIVATION_1(FEATURE_MAP_1)
        POOLED_MAP_1 = self.MAX_POOLING_LAYER_1(ACTIVATED_MAP_1)

        # Pass input through Convolutional Block 2
        FEATURE_MAP_2 = self.CONVOLUTION_LAYER_2(POOLED_MAP_1)
        ACTIVATED_MAP_2 = self.RELU_ACTIVATION_2(FEATURE_MAP_2)
        POOLED_MAP_2 = self.MAX_POOLING_LAYER_2(ACTIVATED_MAP_2)

        # Flatten 4D spatial feature tensor (BATCH_SIZE, 32, 8, 8) into 2D matrix (BATCH_SIZE, 2048)
        FLATTENED_TENSOR = POOLED_MAP_2.view(POOLED_MAP_2.size(0), -1)

        # Pass through Fully Connected Classifier Layers
        DENSE_OUTPUT_1 = self.FULLY_CONNECTED_LAYER_1(FLATTENED_TENSOR)
        ACTIVATED_DENSE_1 = self.RELU_ACTIVATION_3(DENSE_OUTPUT_1)
        FINAL_LOGITS_OUTPUT = self.FULLY_CONNECTED_LAYER_2(ACTIVATED_DENSE_1)

        return FINAL_LOGITS_OUTPUT

# TRAINING & EVALUATION PIPELINE

if __name__ == "__main__":
    # Automatically configure compute device (CUDA GPU if local environment supports it, else CPU)
    COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING COMPUTE DEVICE: {COMPUTE_DEVICE}")

    # Set training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    TOTAL_EPOCHS = 5
    NUMBER_OF_CLASSES = 10

    # Instantiate custom scratch transforms pipeline
    SCRATCH_TRANSFORM_PIPELINE = CUSTOM_COMPOSE_TRANSFORMS([
        CUSTOM_TO_TENSOR(),
        CUSTOM_NORMALIZE(MEAN_TUPLE=(0.4914, 0.4822, 0.4465), STD_TUPLE=(0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 datasets applying our custom transform pipeline
    TRAINING_DATASET = torchvision.datasets.CIFAR10(
        root='./DATA', train=True, download=True, transform=SCRATCH_TRANSFORM_PIPELINE
    )
    TESTING_DATASET = torchvision.datasets.CIFAR10(
        root='./DATA', train=False, download=True, transform=SCRATCH_TRANSFORM_PIPELINE
    )

    # Instantiate PyTorch DataLoader iterators
    TRAIN_DATA_LOADER = torch.utils.data.DataLoader(TRAINING_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    TEST_DATA_LOADER = torch.utils.data.DataLoader(TESTING_DATASET, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize scratch model, loss function, and optimizer
    CNN_MODEL_INSTANCE = FULL_SCRATCH_CNN(NUMBER_OF_CLASSES=NUMBER_OF_CLASSES).to(COMPUTE_DEVICE)
    CUSTOM_LOSS_FUNCTION = SCRATCH_CROSS_ENTROPY_LOSS()
    OPTIMIZER_ADAM = torch.optim.Adam(CNN_MODEL_INSTANCE.parameters(), lr=LEARNING_RATE)

    # TRAINING LOOP
    print("\nSTARTING TRAINING LOOP")
    for EPOCH_INDEX in range(TOTAL_EPOCHS):
        CNN_MODEL_INSTANCE.train()  # Set model to training mode
        RUNNING_LOSS = 0.0

        for BATCH_INDEX, (BATCH_IMAGES, BATCH_LABELS) in enumerate(TRAIN_DATA_LOADER):
            # Transfer batch data tensors to target device (GPU or CPU)
            BATCH_IMAGES = BATCH_IMAGES.to(COMPUTE_DEVICE)
            BATCH_LABELS = BATCH_LABELS.to(COMPUTE_DEVICE)

            # Clear accumulated gradients from previous iteration
            OPTIMIZER_ADAM.zero_grad()

            # Forward pass: compute predictions from model
            PREDICTION_OUTPUTS = CNN_MODEL_INSTANCE(BATCH_IMAGES)

            # Compute loss using our scratch Cross Entropy module
            CALCULATED_LOSS = CUSTOM_LOSS_FUNCTION(PREDICTION_OUTPUTS, BATCH_LABELS)

            # Backward pass: compute parameter gradients via automatic differentiation
            CALCULATED_LOSS.backward()

            # Update layer weights and biases
            OPTIMIZER_ADAM.step()

            RUNNING_LOSS += CALCULATED_LOSS.item()

            # Print intermediate loss statistics every 200 batches
            if (BATCH_INDEX + 1) % 200 == 0:
                AVERAGE_BATCH_LOSS = RUNNING_LOSS / 200
                print(
                    f"EPOCH [{EPOCH_INDEX + 1}/{TOTAL_EPOCHS}] | BATCH [{BATCH_INDEX + 1}/{len(TRAIN_DATA_LOADER)}] | LOSS: {AVERAGE_BATCH_LOSS:.4f}")
                RUNNING_LOSS = 0.0

    # EVALUATION LOOP
    print("\nSTARTING EVALUATION")
    CNN_MODEL_INSTANCE.eval()  # Set model to evaluation mode
    TOTAL_CORRECT_PREDICTIONS = 0
    TOTAL_TEST_SAMPLES = 0

    # Disable gradient computation during testing to reduce memory consumption
    with torch.no_grad():
        for BATCH_IMAGES, BATCH_LABELS in TEST_DATA_LOADER:
            BATCH_IMAGES = BATCH_IMAGES.to(COMPUTE_DEVICE)
            BATCH_LABELS = BATCH_LABELS.to(COMPUTE_DEVICE)

            # Compute predictions on test batch
            PREDICTION_OUTPUTS = CNN_MODEL_INSTANCE(BATCH_IMAGES)

            # Extract highest logit score index per image to get predicted class label
            _, PREDICTED_CLASSES = torch.max(PREDICTION_OUTPUTS, dim=1)

            # Accumulate accuracy counters
            TOTAL_TEST_SAMPLES += BATCH_LABELS.size(0)
            TOTAL_CORRECT_PREDICTIONS += (PREDICTED_CLASSES == BATCH_LABELS).sum().item()

    # Calculate final test accuracy percentage
    FINAL_ACCURACY = (TOTAL_CORRECT_PREDICTIONS / TOTAL_TEST_SAMPLES) * 100.0
    print(f"\nFINAL TEST ACCURACY ON CIFAR-10: {FINAL_ACCURACY:.2f}%")
