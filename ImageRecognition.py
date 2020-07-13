from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

def normalize(x_train, x_test):

    x_train_norm = x_train.astype('float32')
    x_test_norm = x_test.astype('float32')

    x_train_norm = x_train_norm/255.0
    x_test_norm = x_test_norm/255.0

    return x_train_norm, x_test_norm

def plot_performance(history):
    #plot accuracy
    #pyplot.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_classes = 10
y_train = to_categorical(y_train, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)


x_train, x_test = normalize(x_train, x_test)

# Define hyperparameters
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(x_train)//BATCH_SIZE
EPOCHS = 5

model = Sequential()
model.add(Conv2D(32, (FILTER_SIZE, FILTER_SIZE), padding='same', input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
model.add(Conv2D(32, (FILTER_SIZE, FILTER_SIZE), padding='same', input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (FILTER_SIZE, FILTER_SIZE),padding='same', activation = 'relu'))
model.add(Conv2D(64, (FILTER_SIZE, FILTER_SIZE), padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (FILTER_SIZE, FILTER_SIZE),padding='same', activation = 'relu'))
model.add(Conv2D(128, (FILTER_SIZE, FILTER_SIZE),padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

history = model.fit_generator(aug.flow(x_train, y_train, batch_size=BATCH_SIZE), validation_data=(x_test, y_test), steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=2)

score = model.evaluate(x_test, y_test, verbose=1)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))

plot_performance(history)

for i in range(9):
    #defining subplot
    plt.subplot(330+1+i)
    #plot raw pixel data
    plt.imshow(x_train[i])
#plt.show()

