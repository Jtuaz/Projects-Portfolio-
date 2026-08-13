from ultralytics import YOLO # Import ultralytics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Initialize the YOLOv8 model
MODEL = YOLO('yolov8n.pt')

def main():
    # Load the YOLOv8 model
    MODEL = YOLO('yolov8n.pt')

    # Start training
    MODEL.train(
        DATA='C:/BoneFractureYolo8/data.yaml',  # Path to the dataset YAML file
        NUMBER_OF_EPOCHS=100,                             # Number of training epochs
        BATCH_SIZE=32,                               # Batch size (adjust based on system's memory capacity)
        TRAINING_SESSION_NAME='train2',                          # Name for the training session
        NUMNER_OF_WORKERS=2                               # Set to >0 for multiprocessing (adjust based on your system)
    )

    # Evaluate the model on the validation/test dataset to get predictions
    RESULTS = MODEL.val(data='C:/BoneFractureYolo8/data.yaml')

    # Extract ground truth labels and predicted labels
    Y_TRUE = RESULTS['labels']         # True class labels
    Y_PREDICTED = RESULTS['predicted'] # Predicted class labels

    # Calculate accuracy
    ACCURACY = accuracy_score(Y_TRUE, Y_PREDICTED)
    print(f"Model Accuracy: {ACCURACY:.2%}")

    # Generate and plot the confusion matrix
    CONFUSION_MATRIX = confusion_matrix(Y_TRUE, Y_PREDICTED)

    # Set up the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(CONFUSION_MATRIX, annot=True, fmt="d", cmap="Blues",
                xticklabels=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'shoulder fracture', 'wrist positive'],
                yticklabels=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'shoulder fracture', 'wrist positive'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig('D:/MachineVision/confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    main()
