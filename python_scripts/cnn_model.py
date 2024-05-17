import numpy as np
import keras
import cv2
import pandas as pd
import matplotlib.pyplot as plt

mnist_test_dir = 'archive/mnist_test_normalized.csv'
mnist_train_dir = 'archive/mnist_train_normalized.csv'

train_data = pd.read_csv(mnist_train_dir)
test_data = pd.read_csv(mnist_test_dir)

x_train = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1)
y_train = train_data['label'].values

x_test = test_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1)
y_test = test_data['label'].values

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.AveragePooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.AveragePooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.AveragePooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10)
loss, accuracy = model.evaluate(x_test,y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

print("------")

img = cv2.imread("zero.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = np.invert(np.array([img]))
img = img.reshape(1, 28, 28, 1)

prediction = model.predict(img)
predicted_number = np.argmax(prediction)
print("Predicted number:", predicted_number)

img2 = cv2.imread("one.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (28, 28))
img2 = np.invert(np.array([img2]))
img2 = img2.reshape(1, 28, 28, 1)

prediction2 = model.predict(img2)
predicted_number2 = np.argmax(prediction2)
print("Predicted number:", predicted_number2)

img22 = cv2.imread("three.jpg", cv2.IMREAD_GRAYSCALE)
img22 = cv2.resize(img22, (28, 28))
img22 = np.invert(np.array([img22]))
img22 = img22.reshape(1, 28, 28, 1)

prediction22 = model.predict(img22)
predicted_number22 = np.argmax(prediction22)
print("Predicted number:", predicted_number22)

img222 = cv2.imread("six.jpg", cv2.IMREAD_GRAYSCALE)
img222 = cv2.resize(img222, (28, 28))
img222 = np.invert(np.array([img22]))
img222 = img222.reshape(1, 28, 28, 1)

prediction222 = model.predict(img222)
predicted_number222 = np.argmax(prediction222)
print("Predicted number:", predicted_number222)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()

model.save('cnn.h5')
