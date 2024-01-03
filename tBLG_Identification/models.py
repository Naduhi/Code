import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class GANModel:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

        self.generator = self.Generator_GAN()
        self.discriminator = self.Discriminator_GAN()

    def Discriminator_GAN(self):

        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.feature_dim)))
        model.add(layers.Reshape([self.feature_dim, 1]))

        model.add(layers.Conv1D(kernel_size=16, filters=256, activation='leaky_relu'))
        model.add(layers.MaxPool1D())
        model.add(layers.Dropout(0.02))
        model.add(layers.Conv1D(kernel_size=16, filters=128))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(layers.MaxPool1D())
        model.add(layers.Dropout(0.02))

        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dense(1))
        model.compile()
        return model

    def Generator_GAN(self):

        model = tf.keras.Sequential()
        model.add(layers.Input((self.feature_dim)))

        model.add(Dense(128, activation='leaky_relu'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        model.add(layers.Dense(256))
        model.add(LeakyReLU(alpha=0.01))
        model.add(layers.Dense(self.feature_dim))
        model.compile()

        return model

    def generate_data(self, num_synthetic_to_gen=1):
        noise_input = tf.random.normal([num_synthetic_to_gen, self.feature_dim])
        predictions = self.generator(noise_input, training=False)
        return  predictions.numpy()

    def load_generator(self, checkpoint_path):
        self.generator = tf.keras.models.load_model(checkpoint_path)
        self.generator.compile()



class ClassificationModels:
    def __init__(self, xtrain):
        self.xtrain = xtrain

    def Classifier_CNN(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.xtrain.shape[1])))
        model.add(layers.Reshape([self.xtrain.shape[1], 1]))

        model.add(layers.Conv1D(kernel_size=3, strides=2, filters=32))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.01))

        model.add(layers.Conv1D(kernel_size=3, filters=64, strides=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.01))

        model.add(layers.Conv1D(kernel_size=3, filters=128, strides=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.01))

        model.add(layers.Conv1D(kernel_size=3, filters=256, strides=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.01))

        model.add(layers.Flatten())
        model.add(layers.Dense(4, activation='softmax'))
        model.compile()
        return model

    def Classifier_ANN(self):
        return MLPClassifier(hidden_layer_sizes=(100, 200), alpha=0.01, max_iter=1000, random_state=0)

    def Random_forest(self):
        return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=0)

    def Decision_tree(self):
        return DecisionTreeClassifier(max_depth=9, min_samples_split=2, min_samples_leaf=3, splitter="best",
                                      random_state=0)

    def Support_vector_machine(self):
        return SVC(gamma="auto", kernel="linear", C=3, probability=True)

    def K_nearest_neighbor(self):
        return KNeighborsClassifier(n_neighbors=2, leaf_size=3, weights="distance")

    def Logistic_regression(self):
        return LogisticRegression(max_iter=10000, random_state=0)
