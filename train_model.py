import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(augment=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalization:
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_trainn, x_val, y_trainn, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Reshape for CNN
    img_width, img_height, num_channels = 28, 28, 1  # MNIST images are 28x28 grayscale
    x_train_cnn = x_trainn.reshape(x_trainn.shape[0], img_width, img_height, num_channels)
    x_val_cnn = x_val.reshape(x_val.shape[0], img_width, img_height, num_channels)
    x_test_cnn = x_test.reshape(x_test.shape[0], img_width, img_height, num_channels)

    # Flatten for ANN
    x_train_ann = x_trainn.reshape(x_trainn.shape[0], 784)
    x_val_ann = x_val.reshape(x_val.shape[0], 784)
    x_test_ann = x_test.reshape(x_test.shape[0], 784)

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=15, #Increased rotation
            width_shift_range=0.15, #Increased shift
            height_shift_range=0.15, #Increased shift
            zoom_range=0.15, #Increased zoom
            fill_mode='constant',
            cval=0,
            preprocessing_function=add_random_background,
            horizontal_flip=False,  # Or False, depending on your digits
            vertical_flip=False,
        )
        datagen.fit(x_train_cnn)
        train_generator = datagen.flow(x_train_cnn, y_trainn, batch_size=32)  # Correctly use flow

        return (x_train_ann, x_val_ann, x_test_ann, train_generator, x_val_cnn, x_test_cnn, y_trainn, y_val, y_test)  # Return train_generator
    else:
        return (x_train_ann, x_val_ann, x_test_ann, x_train_cnn, x_val_cnn, x_test_cnn, y_trainn, y_val, y_test)


def add_random_background(image):
    """Randomly inverts the image with 50% probability."""
    if np.random.random() < 0.5:
        return 1 - image  # Invert the image
    else:
        return image

def build_model(model_type="ANN", hidden_layers_neurons=[128, 64], dropout_rate=0.2, l2_lambda=0.001):
    logging.info(f"Building {model_type} model...")

    if model_type == "ANN":
        model = Sequential()
        model.add(Input(shape=(784,)))
        model.add(Dense(units=hidden_layers_neurons[0], activation='relu',
                        kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for neurons in hidden_layers_neurons[1:]:
            model.add(Dense(units=neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=10, activation='softmax'))
        logging.info(f"ANN model built with hidden layers: {hidden_layers_neurons}, dropout: {dropout_rate}, L2: {l2_lambda}")
        return model

    elif model_type == "CNN":
        img_width, img_height, num_channels = 28, 28, 1
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, num_channels), kernel_regularizer=regularizers.l2(l2_lambda)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(10, activation='softmax')
        ])
        logging.info(f"CNN model built with dropout: {dropout_rate}, L2: {l2_lambda}")
        return model

    else:
        raise ValueError("Invalid model_type. Choose 'ANN' or 'CNN'.")



def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=64, patience=5, learning_rate=0.001, use_generator=False): #Increased epochs and batch size
    logging.info(f"Training model with epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}")
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    if use_generator:
        history = model.fit(x_train,  # x_train is now a generator
                            steps_per_epoch=len(x_train),  # Number of batches per epoch
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping],
                            verbose=1)
    else:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val), callbacks=[early_stopping])

    logging.info("Model training complete.")
    return history, model


def plot_training_history(history):
    logging.info("Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
    logging.info("Training history plotted.")


def evaluate_model(model, x_test, y_test, model_type="ANN"):
    logging.info(f"Evaluating {model_type} model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    logging.info(f"Test Loss: {test_loss*100:.4f} %")
    logging.info(f"Test Accuracy: {test_accuracy*100:.4f} %")

    y_pred = np.argmax(model.predict(x_test), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=range(10), yticklabels=range(10), cbar=True, annot_kws={"size": 7})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

    logging.info("\nClassification Report:\n", classification_report(y_test, y_pred))

    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=np.arange(10))
    n_classes = y_test_bin.shape[1]

    y_prob = model.predict(x_test)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(8, 6))
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc="best", fontsize = "small")
    plt.show()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="best", fontsize = "small")
    plt.show()

    logging.info(f"{model_type} model evaluation complete.")


def tune_hyperparameters(x_train, y_train, x_val, y_val, model_type):
    logging.info(f"Tuning hyperparameters for {model_type} model...")
    tuning_results = []
    best_accuracy = 0.0
    best_params = {}

    learning_rates = [0.0005, 0.001, 0.002] #Added more learning rates
    dropout_rates = [0.1, 0.2, 0.3] #Added more dropout rates
    hidden_layer_sizes = [[128], [256], [128, 64], [256, 128]] #Added more hidden layer sizes
    l2_lambdas = [0.0005, 0.001, 0.002] #Added more L2 lambdas

    for lr in learning_rates:
        for dr in dropout_rates:
            for hl in hidden_layer_sizes:
                for ll in l2_lambdas:
                    logging.info(f"Trying model={model_type}, learning rate={lr}, dropoutrates={dr}, hidden_layer_sizes={hl}, l2_lambdas ={ll}")

                    model = build_model(model_type=model_type, hidden_layers_neurons=hl, dropout_rate=dr, l2_lambda=ll)

                    if model_type == "ANN":
                        train_data = x_train
                        val_data = x_val
                    elif model_type == "CNN":
                        train_data = x_train
                        val_data = x_val
                    else:
                        raise ValueError("Invalid model_type.")

                    history, _ = train_model(model, train_data, y_train, val_data, y_val, epochs=50, batch_size=64, #Increased epochs and batch size
                                              patience=5, learning_rate=lr)
                    val_accuracy = history.history['val_accuracy'][-1]

                    tuning_results.append({'model_type': model_type, 'learning_rate': lr, 'dropout_rate': dr,
                                           'hidden_layers': hl, 'l2_lambda': ll, 'val_accuracy': val_accuracy})

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = {'model_type': model_type, 'learning_rate': lr, 'dropout_rate': dr,
                                       'hidden_layers': hl, 'l2_lambda': ll}

    logging.info(f"Best Hyperparameters: {best_params}")
    logging.info(f"Hyperparameter tuning for {model_type} complete.")
    return best_params, tuning_results

def get_otsu_threshold(image_array):
    """Calculates Otsu's threshold for a grayscale image."""
    # Flatten the image array
    pixels = image_array.flatten()
    
    # Calculate histogram
    histogram, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    
    total_pixels = len(pixels)
    
    # Initialize variables
    max_variance = 0
    threshold = 0
    
    sum_background = 0
    weight_background = 0
    
    # Iterate over all possible threshold values (0 to 255)
    for t in range(256):
        # Update background and foreground statistics
        weight_background += histogram[t]
        
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        
        if weight_foreground == 0:
            break
        
        sum_background += t * histogram[t]
        
        # Calculate means
        mean_background = sum_background / weight_background
        mean_foreground = (np.sum(np.arange(t + 1, 256) * histogram[t + 1:])) / weight_foreground
        
        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Update threshold if variance is larger
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = t
    
    return threshold

if __name__ == '__main__':
    # Load and preprocess data
    x_train_ann, x_val_ann, x_test_ann, train_generator, x_val_cnn, x_test_cnn, y_train, y_val, y_test = load_and_preprocess_data(augment=True)

    # Build and train ANN model
    logging.info("Training ANN Model...")
    ann_model = build_model(model_type="ANN", hidden_layers_neurons=[256, 128], dropout_rate=0.2, l2_lambda=0.001)
    ann_history, ann_trained_model = train_model(ann_model, x_train_ann, y_train, x_val_ann, y_val, epochs=50, batch_size=64, patience=5, learning_rate=0.001) #Increased epochs and batch size

    # Plot ANN training history
    logging.info("ANN Model Training Performance")
    plot_training_history(ann_history)

    # Evaluate ANN
    logging.info("Evaluating ANN Model...")
    evaluate_model(ann_trained_model, x_test_ann, y_test, model_type="ANN")

    # Build and train CNN model
    logging.info("Training CNN Model...")
    cnn_model = build_model(model_type="CNN", dropout_rate=0.2, l2_lambda=0.001)
    cnn_history, cnn_trained_model = train_model(cnn_model, train_generator, y_train, x_val_cnn, y_val, epochs=50, batch_size=64, patience=5, learning_rate=0.001, use_generator=True) #Increased epochs and batch size

    # Plot CNN training history
    logging.info("CNN Model Training Performance")
    plot_training_history(cnn_history)

    # Evaluate CNN
    logging.info("Evaluating CNN Model...")
    evaluate_model(cnn_trained_model, x_test_cnn, y_test, model_type="CNN")

    # Tabular comparison of tuning results
    logging.info('Before Tuning ANN')
    best_params_ann, tuning_results_ann = tune_hyperparameters(x_train_ann, y_train, x_val_ann, y_val, model_type="ANN")

    logging.info('Before Tuning CNN')
    best_params_cnn, tuning_results_cnn = tune_hyperparameters(x_train_cnn, y_train, x_val_cnn, y_val, model_type="CNN")

    df_ann = pd.DataFrame(tuning_results_ann)
    logging.info("\nTuning Results Table for ANN:\n")
    logging.info(df_ann)

    df_cnn = pd.DataFrame(tuning_results_cnn)
    logging.info("\nTuning Results Table for CNN:\n")
    logging.info(df_cnn)

    # Basic Analysis
    logging.info("\nBasic Analysis for ANN:")
    logging.info(f"Number of trials: {len(df_ann)}")
    logging.info(f"Best Validation Accuracy: {df_ann['val_accuracy'].max():.4f}")

    logging.info("\nBasic Analysis for CNN:")
    logging.info(f"Number of trials: {len(df_cnn)}")
    logging.info(f"Best Validation Accuracy: {df_cnn['val_accuracy'].max():.4f}")