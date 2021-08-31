import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from load_data import load_data
from params import path_to_data, data_files


class Trainer:

    def __init__(self, x, y):
        self.model = None
        self.x = x
        self.y = y

    def base_model(self):

        #DataFlair - Initialize the Multi Layer Perceptron Classifier
        self.model = MLPClassifier(alpha=0.01,
                              batch_size=256,
                              epsilon=1e-08,
                              hidden_layer_sizes=(300, ),
                              learning_rate='adaptive',
                              max_iter=500)

    def run(self):

        self.model.fit(self.x, self.y)

    def evaluate(self, x_test, y_test):

        y_pred = self.model.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)
        return acc_score

if __name__ == "__main__":
    x, y = load_data(path_to_data, data_files)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x),
                                                    y,
                                                    test_size=0.2,
                                                    random_state=9)


    # Train and save model, locally and
    trainer = Trainer(x=x_train, y=y_train)
    trainer.base_model()

    # trainer.set_experiment_name('xp2') - mlflow
    trainer.run()
    score = trainer.evaluate(x_test, y_test)
    print(f"accuracy score: {score}")
