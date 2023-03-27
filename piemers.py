import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense #sasaista slāņus

from tensorflow.keras.datasets import mnist
#MNIST Modified National Institute of Standards and Technology
#x_train - apmācības parauga cipara attēli
#y_train - apmācības parauga ciparu vērtību vektors
#x_test - testa parauga cipara attēli
#y_test - testa parauga ciparu vērtību vektors

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.figure(figsize=(10, 5))

#izdruka
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28,1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())
# x vektors uz diapazonu no 0 līdz 1
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(x_train, y_train_cat, batchsize=32, epochs=10, validation_split=0.2)
#parbauda testa kopu ar metodi evaluate
model.evaluate(x_test, y_test_cat)

#atpazīst kādu no testa attēliem
n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)

#izdruka lielako skaitli
print(np.argmax(res))
#uz ekrana izdruka attēlu
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()
#nepareizi rez 0 līdz 9
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred.shape)
print(pred[:20])
print(y_test[:20])
#masku testam True/False
mask = pred == y_test
print(mask[:10])
x_false = x_test[~mask]
y_false = x_test[~mask]
print(x_false.shape)
#pirmos 5 uz ekrāna
for i in range(5):
    print("Tīkla vērtība:"+str(y_test[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()