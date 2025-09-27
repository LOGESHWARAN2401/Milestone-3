#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[2]:


# Paths
path = r"E:\milestone 3\Train"
labelFile = r"E:\milestone 3\train.csv"

# Parameters
batch_size_val = 32
epochs_val = 10
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2


# In[10]:


import os
import cv2
import numpy as np

# Fix the path
path = r"E:\milestone 3\Train"

images = []
classNo = []

# List all class folders
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)

# Loop through each class folder
for count, folder in enumerate(myList):
    folder_path = os.path.join(path, folder)
    myPicList = os.listdir(folder_path)

    for y in myPicList:
        curImg = cv2.imread(os.path.join(folder_path, y))
        if curImg is not None:  # check image is valid
            curImg = cv2.resize(curImg, (32, 32))   # resize to 32x32
            images.append(curImg)
            classNo.append(count)  # label is folder index

    print(count, end=" ")

print("\nImport Completed!")

images = np.array(images)
classNo = np.array(classNo)

print("Images shape:", images.shape)
print("Labels shape:", classNo.shape)


# In[9]:


get_ipython().system('pip install opencv-python')


# In[15]:


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   # ✅ import added

# Example ratios (set them before using)
testRatio = 0.2         # 20% for test
validationRatio = 0.2   # 20% of remaining train for validation

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    images, classNo, test_size=testRatio, random_state=42, stratify=classNo
)

# Split training further into train/validation
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=validationRatio, random_state=42, stratify=y_train
)

print("Data Shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# ✅ Load label file (make sure you set the correct path!)
labelFile = r"E:\milestone 3\labels.csv"   # <-- change this to your actual labels file path
data = pd.read_csv(labelFile)
print("Label Data shape:", data.shape)


# In[14]:


get_ipython().system('pip install scikit-learn')


# In[16]:


def preprocessing(img):
    img = img / 255.0   # normalize RGB values between 0-1
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Data Augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# One-hot encode labels
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# In[17]:


def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape=imageDimesions, activation='relu'))
    model.add(Conv2D(60, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[18]:


model = myModel()
print(model.summary())

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)


# In[19]:


# Training & Validation Curves

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()


# In[20]:


# Evaluate model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


# In[23]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded y_test to integers
y_true = np.argmax(y_test, axis=1)

# Metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average="weighted")
recall = recall_score(y_true, y_pred_classes, average="weighted")
f1 = f1_score(y_true, y_pred_classes, average="weighted")

print("\n Model Performance Metrics")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n Detailed Classification Report:\n")
print(classification_report(y_true, y_pred_classes, digits=4))


# In[25]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.show()


# In[26]:


# Save Model

model.save("model_rgb.h5")
print("Model saved as model_rgb.h5")


# In[ ]:




