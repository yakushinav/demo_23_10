#Пример 4. Простая нейронная сеть
# Для работы нужны библиотеки pandas, sklearn, tensorflow, matplotlib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

DF = pd.read_csv('input.csv', delimiter=";")

target = DF.pop('target')
validation_size = 0.20
seed = 7
scoring = 'accuracy'
x_train, x_test, y_train, y_test = train_test_split(DF, target, test_size=validation_size, random_state=seed)

# Предобработаем данные (это массивы Numpy)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Зарезервируем 4000 примеров для валидации
x_val = x_train[-4000:]
y_val = y_train[-4000:]
x_train = x_train[:-4000]
y_train = y_train[:-4000]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
	keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

# Укажем конфигурацию обучения (оптимизатор, функция потерь, метрики)
model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                  # Минимизируемая функция потерь
                  loss='mean_squared_error',
                  metrics=['mae'])

# Обучим модель разбив данные на "пакеты"
# размером "batch_size", и последовательно итерируя
# весь датасет заданное количество "эпох"
history = model.fit(x_train, y_train,
                        batch_size=2,
                        epochs=3,
                        # Мы передаем валидационные данные для
                        # мониторинга потерь и метрик на этих данных
                        # в конце каждой эпохи
                        validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
predictions = model.predict(x_test[:5])
print(x_test[:5],"\n", predictions)

print(history.history)
plt.plot(history.history['val_mae'])
print(history.history['val_mae'][-1])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend('mean_squared_error', loc='upper left')
plt.show()