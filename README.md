# Brain Tumor Detection using EfficientNetB0 and TensorFlow

This project implements a deep learning-based approach for brain tumor detection using the EfficientNetB0 model. It includes training with advanced data augmentation, fine-tuning, model evaluation (ROC curves, confusion matrices), and interpretability with Grad-CAM.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grad-CAM (Interpretability)](#grad-cam-interpretability)
- [Results](#results)

## Installation

To run this project, install the following dependencies:

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python
```

## Dataset

The dataset used is organized into two directories:
- **Training**: Contains brain tumor images for training and validation.
- **Testing**: Contains brain tumor images for model testing.

Directory structure:

```bash
/brain_tumor_dataset
    ├── Training/
    └── Testing/
```

You can replace the dataset path in the code:

```python
data_path = "/path/to/brain_tumor_dataset"
```

## Model Architecture

The model is built on top of the pre-trained EfficientNetB0 architecture, leveraging transfer learning and fine-tuning. Additional layers are added for better feature extraction and classification.

```python
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```

## Training

We utilize advanced data augmentation techniques like horizontal/vertical flips, rotations, zoom, brightness adjustments, etc., to increase model generalization.

Training configuration includes:
- **Optimizer**: Adam
- **Learning Rate**: 0.00001 for fine-tuning
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy

```python
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Callbacks
- EarlyStopping: Stops training when validation loss stops improving.
- ReduceLROnPlateau: Reduces learning rate when the validation loss plateaus.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
```

Run the training process:

```python
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[early_stopping, reduce_lr]
)
```

## Evaluation

### Validation Set

After training, the model is evaluated on the validation set:

```python
val_loss, val_accuracy = model.evaluate(val_gen)
```

### Test Set

The model is evaluated on the unseen test set:

```python
test_loss, test_accuracy = model.evaluate(test_gen)
```

### ROC Curves

The model's performance is further assessed by plotting ROC curves and calculating the Area Under the Curve (AUC) for each class.

```python
fpr, tpr, roc_auc = compute_roc_auc(y_true, y_pred, class_labels)
plot_roc_curves(fpr, tpr, roc_auc, class_labels)
```

### Confusion Matrix

The confusion matrix provides insight into the model's classification performance.

```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
```

## Grad-CAM (Interpretability)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which parts of the image contribute most to the model's decision.

To generate and display Grad-CAM:

```python
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv")
display_gradcam(img_path, heatmap)
```

Example image with Grad-CAM applied:

![Grad-CAM](cam.jpg)

## Results

- **Validation Accuracy**: `XX%`
- **Test Accuracy**: `XX%`

### Example ROC Curves:

![ROC Curves](Roc_curves.png)

### Confusion Matrices:

Validation Set             |  Test Set
:-------------------------:|:-------------------------:
![Validation Confusion Matrix](val_confusion_matrix.png)  |  ![Test Confusion Matrix](test_confusion_matrix.png)

---
