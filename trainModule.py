import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


path = 'dataset'
X_TRAIN = []
Y_TRAIN = []

label_dict = {'huu': [1,0], 'mask': [0,1]}
# neeus them label thi tang so luong len der ko loi , vd 3 nguoi thi 100 ,010,001 
def getData(path, X_TRAIN):
    for whatever in os.listdir(path):
        whatever_path = os.path.join(path, whatever)
        lst_filename_path = []

        for filename in os.listdir(whatever_path):
            fileName_path = os.path.join(whatever_path, filename)
            label = fileName_path.split('\\')[1]

            image = cv2.imread(fileName_path, cv2.IMREAD_COLOR)
            # Resize the image to (32, 32)
            image = cv2.resize(image, (32, 32))
            lst_filename_path.append((image, label_dict[label]))
        X_TRAIN.extend(lst_filename_path)
    return X_TRAIN

X_data = getData(path, X_TRAIN)
np.random.shuffle(X_data)
np.random.shuffle(X_data)
np.random.shuffle(X_data)
# Chuyển đổi dữ liệu đầu vào
X_array = np.array([x[0] for x in X_data])
X = X_array.astype('float32') / 255.0

# Chuẩn bị dữ liệu đầu ra
Y = np.array([x[1] for x in X_data])

 

# Xây dựng mô hình
model_training_first = models.Sequential()

model_training_first.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model_training_first.add(layers.MaxPool2D((2, 2)))
model_training_first.add(layers.Dropout(0.15))

model_training_first.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_training_first.add(layers.MaxPool2D((2, 2)))
model_training_first.add(layers.Dropout(0.15))

model_training_first.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_training_first.add(layers.MaxPool2D((2, 2)))
model_training_first.add(layers.Dropout(0.15))

model_training_first.add(layers.Flatten())
model_training_first.add(layers.Dense(1000, activation='relu'))
model_training_first.add(layers.Dense(256, activation='relu'))
model_training_first.add(layers.Dense(2, activation='softmax'))

model_training_first.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_training_first.summary()

# Huấn luyện mô hình
model_training_first.fit(X , Y, epochs=10 )


# Save the model in the recommended format
model_training_first.save('trained_model.keras')

