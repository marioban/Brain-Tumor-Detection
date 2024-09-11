import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image

# Load and preprocess the dataset with advanced augmentation
def load_and_preprocess_data(data_path):
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        os.path.join(data_path, 'Training'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical', 
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(data_path, 'Training'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',  
        subset='validation'
    )

    return train_generator, validation_generator

# Function to load the test data without augmentation (only rescaling)
def load_test_data(test_data_path):
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Ensure the test set is not shuffled
    )
    return test_generator

data_path = "/Users/marioban/Desktop/Faks/Computer Vision/brain_tumor_dataset"
test_data_path = os.path.join(data_path, 'Testing')  # Define the test data path

# Load training, validation, and test sets
train_gen, val_gen = load_and_preprocess_data(data_path)
test_gen = load_test_data(test_data_path)

# Determine the number of classes
num_classes = len(train_gen.class_indices)

# Define the model using EfficientNet for better performance
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Use num_classes instead of hardcoded value

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {test_accuracy}")

# Function to compute ROC curve and AUC
def compute_roc_auc(y_true, y_pred, class_labels):
    n_classes = len(class_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc

# Function to plot ROC curves
def plot_roc_curves(fpr, tpr, roc_auc, class_labels):
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

    # Plot micro-average ROC curve
    plt.plot(fpr['micro'], tpr['micro'],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Compute and plot ROC curves for validation set
val_gen.reset()
y_true_val = val_gen.classes
y_pred_val = model.predict(val_gen)
class_labels = list(val_gen.class_indices.keys())

fpr_val, tpr_val, roc_auc_val = compute_roc_auc(y_true_val, y_pred_val, class_labels)
plot_roc_curves(fpr_val, tpr_val, roc_auc_val, class_labels)

# Compute and plot ROC curves for test set
test_gen.reset()
y_true_test = test_gen.classes
y_pred_test = model.predict(test_gen)

fpr_test, tpr_test, roc_auc_test = compute_roc_auc(y_true_test, y_pred_test, class_labels)
plot_roc_curves(fpr_test, tpr_test, roc_auc_test, class_labels)

# Confusion matrix for validation set
y_pred_classes_val = np.argmax(y_pred_val, axis=1)

cm_val = confusion_matrix(y_true_val, y_pred_classes_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Validation Set Confusion Matrix')
plt.show()

# Generate a classification report for the validation set
print(classification_report(y_true_val, y_pred_classes_val, target_names=val_gen.class_indices.keys()))

# Confusion matrix for test set
y_pred_classes_test = np.argmax(y_pred_test, axis=1)

cm_test = confusion_matrix(y_true_test, y_pred_classes_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Set Confusion Matrix')
plt.show()

# Generate a classification report for the test set
print(classification_report(y_true_test, y_pred_classes_test, target_names=test_gen.class_indices.keys()))

# Grad-CAM for model interpretability
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

# Get the file paths of the test images for Grad-CAM
test_image_paths = test_gen.filepaths

# Example: Use the first test image for Grad-CAM
img_path = test_image_paths[0]  # Replace with the index of the desired image
img_array = get_img_array(img_path, size=(224, 224))
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv")  # Confirm from model.summary()
display_gradcam(img_path, heatmap)

# Display model summary to check the last convolutional layer for Grad-CAM
model.summary()