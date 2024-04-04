import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class GestureRecognitionModel:
    def __init__(self, dataset_path, learning_rate=0.0001, epochs=5, batch_size=16):
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.trained_model = None

    def load_dataset(self, test_size=0.2, random_seed=42):
        X_data, y_data = [], []

        train_path = os.path.join(self.dataset_path, 'train')
        test_path = os.path.join(self.dataset_path, 'test')

        label_mapping = {}
        for i, label in enumerate(os.listdir(train_path)):
            label_mapping[label] = i

        for label in os.listdir(train_path):
            train_label_path = os.path.join(train_path, label)
            test_label_path = os.path.join(test_path, label)

            train_image_files = os.listdir(train_label_path)
            test_image_files = os.listdir(test_label_path)

            num_train_images = len(train_image_files)
            num_test_images = int(test_size * num_train_images)

            np.random.seed(random_seed)
            train_image_files = np.random.permutation(train_image_files)

            train_files = train_image_files[num_test_images:]
            for filename in train_files:
                img_path = os.path.join(train_label_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128,128))
                img = np.expand_dims(img, axis=-1)
                X_data.append(img)
                y_data.append(label_mapping[label])

        combined_data = list(zip(X_data, y_data))
        np.random.shuffle(combined_data)
        X_data, y_data = zip(*combined_data)

        X_data = np.array(X_data) / 255.0
        y_data = np.array(y_data)

        return X_data, y_data

    def initialize_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(8, activation='softmax'))
        return model

    def train(self, X_train, y_train):
        model = self.initialize_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        split = int(len(X_train) * 0.8)
        X_train_split, X_val_split = X_train[:split], X_train[split:]
        y_train_split, y_val_split = y_train[:split], y_train[split:]

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        class PrintEpochStats(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1}/{self.params['epochs']} - "
                    f"Loss: {logs['loss']:.4f} - "
                    f"Accuracy: {logs['accuracy']:.4f} ")

        history = model.fit(
            datagen.flow(X_train_split, y_train_split, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(X_val_split, y_val_split),
            callbacks=[early_stopping, PrintEpochStats()],
            verbose=2
        )

        self.trained_model = model

        return history

    def evaluate_and_save(self, X_test, y_test, model_save_path):
        test_loss, test_accuracy = self.trained_model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


        y_pred_probs = self.trained_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        accuracy = np.mean(y_pred == y_test)

        num_classes = len(np.unique(y_test))
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(num_classes):
            true_positives = np.sum((y_pred == i) & (y_test == i))
            false_positives = np.sum((y_pred == i) & (y_test != i))
            false_negatives = np.sum((y_pred != i) & (y_test == i))

            precision[i] = true_positives / max((true_positives + false_positives), 1)
            recall[i] = true_positives / max((true_positives + false_negatives), 1)
            f1[i] = 2 * (precision[i] * recall[i]) / max((precision[i] + recall[i]), 1)

            for j in range(num_classes):
                confusion_mat[i, j] = np.sum((y_pred == j) & (y_test == i))

        print(f'Precision: {np.mean(precision)}')
        print(f'Recall: {np.mean(recall)}')
        print(f'F1 Score: {np.mean(f1)}')
        print('Confusion Matrix:')
        print(confusion_mat)

        self.trained_model.save(model_save_path)
        print(f'Model saved at: {model_save_path}')

dataset_path = "dataset/data"

gesture_model = GestureRecognitionModel(dataset_path)
X_train, y_train = gesture_model.load_dataset(test_size=0.2, random_seed=42)
history = gesture_model.train(X_train, y_train)

X_test, y_test = gesture_model.load_dataset(test_size=0.2, random_seed=42)
model_save_path = "trained_model/model.h5"
gesture_model.evaluate_and_save(X_test, y_test, model_save_path)
