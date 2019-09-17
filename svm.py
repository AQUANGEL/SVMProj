import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cvxopt import matrix as cvxopt_matrix, solvers as cvxopt_solvers
from sklearn.datasets import load_iris


class SVM:
    def __init__(self):
        self.optimal = False
        self.colors = {1: 'b', -1: 'r'}

    # train
    def train(self, features_matrix, classes_vector):
        self.features_matrix = features_matrix
        self.classes_vector = classes_vector

        if len(features_matrix[0]) <= 3:
            self.figure = plt.figure()
            if len(features_matrix[0]) == 2:
                self.axis = self.figure.add_subplot(1, 1, 1)
            else:
                self.axis = Axes3D(self.figure)

        #  my goal is to take the features matrix and the classes vector and manipulate them in order to solve the
        # dual optimization problem using cvxopt qp method, for that I need to calc an equation\function of the form:
        # min(0.5xTPx + aTx) such that Gx<=h Ax=b

        # where my original equation\function is:
        # min(w,b)(max(a)(L(w,b,a))) = max(a)(min(w,b)L(w,b,a))
        # that is the result of the lagrangian function\equation developed of the following circumstances
        # L(w,b,a) = 0.5 * ||w|| ^ 2 - sigma(i to r)(ai[yi * (wT dot xi + b) - 1])
        # because i want to find the max margin which is given by max(2/||w||)
        # or by min(0.5 * ||w|| ^ 2) with the constraints that for each features vector (aka sigma i to r)
        # y1 * (wT dot xi + b) >= 1

        # that Lagrangian equation\function derivatives get me the following:
        # w = sigma(i to r)(ai * yi * xi)
        # 0 = sigma(i to r)(ai * yi)

        # and eventually : (j's are from replacing w and b with their derivatives values compared to zero)
        # max(a) = sigma(i to r)(ai) - 0.5 * sigma(i, j to r)(ai * aj * yi * yj * xi * xj)
        # such that ai >= 0 and sigma(i to r)(ai * yi) = 0

        # in order to get all that to fit the qp function I'm constructing a matrix M out of yi * yj * (xi * xj)
        # which is actually a multiplication of matrix yX and it's transpose (where y is the classes vector and
        # X is the features matrix)

        # thus getting:
        # max(a) = sigma(i to r)(ai) - 0.5 * aT * M * a
        # and by multiplying it by -1 I'm getting:
        # min(a) = 0.5 * aT * M * a - 1T * a
        # such that -ai <= 0 and yTa = 0

        # where:
        # a is the vector of lagrangian multipliers
        # y is the classes vector
        # M is the matrix as developed above
        # and T stands fot Transpose

        # now I've explained the required manipulation of the date and it's time to write it in code

        # rows = number of vectors
        # cols = number of features
        rows, cols = features_matrix.shape
        classes_matrix = classes_vector.reshape(-1, 1).dot(1.)  # 1. to get float type
        matrix_yX = features_matrix * classes_matrix
        matrix_M = np.matmul(matrix_yX, matrix_yX.T)

        # Converting into cvxopt format
        P = cvxopt_matrix(matrix_M)
        q = cvxopt_matrix(-np.ones((rows, 1)))  # the -1T matrix
        G = cvxopt_matrix(-np.eye(rows))  # I matrix
        h = cvxopt_matrix(np.zeros(rows))  # the constraints vector for -ai <= 0
        A = cvxopt_matrix(classes_matrix.reshape(1, -1))  # the classes vector for yTa = 0
        b = cvxopt_matrix(np.zeros(1))  # the scalar constraint for yTa = 0 (the 0 in the equation)

        res = cvxopt_solvers.qp(P, q, G, h, A, b)

        if res['status'] == 'optimal':
            self.optimal = True
            alphas_vector = np.array(res['x'])

            # vector w in vectorized form - the Holy Grail, aka the decision boundary
            # w = sigma(i to r)(ai * yi * xi)
            self.vector_w = np.matmul((classes_matrix * alphas_vector).T, features_matrix).reshape(-1, 1)

            # Selecting the set of indices S corresponding to non zero parameters, aka the locations of the
            # support vectors(the features vectors that provide the largest margin)
            # following the constraint -ai <= 0 which translates back to a >= 0
            support_vectors_indices = (alphas_vector > 1e-4).flatten()

            # Computing the svm bias
            # every satisfying support vector will form: ySV(xSV * w + b) = 1
            # multiplying that by ySV (which is 1 ot -1) will result in:
            # ySV^2 * (xSV * w + b) = ySV, ySV ^ 2 = 1 so
            # b = ySV - (xSV * w)
            bias = classes_matrix[support_vectors_indices] - np.dot(features_matrix[support_vectors_indices],
                                                                         self.vector_w)

            self.support_vectors = features_matrix[support_vectors_indices]

            self.bias = bias[0]

        return self.optimal

    def predict(self, features_vector):
        # sign( x.w+b ) => we get the sign from np.sign which return 1 || 0 || -1
        # np.dot - calc the scalar multiplication of vector x = features with vector w - the normal vector to the
        # decision boundary, w start from the origin
        # b is the bias of decision boundary (aka the hyper plane) since it won't always start at the origin
        if self.optimal:
            classification = np.sign(np.dot(np.array(features_vector), self.vector_w) + self.bias)
            if classification != 0:
                if len(features_vector) == 2:
                    self.axis.scatter(features_vector[0], features_vector[1], s=200, marker='*',
                                      c=self.colors[classification[0]])
                elif len(features_vector) == 3:
                    self.axis.scatter(features_vector[0], features_vector[1], features_vector[2],
                                      s=200, marker='*', c=self.colors[classification[0]])

            return classification
        else:
            return 0

    def display(self):
        if self.optimal and len(self.features_matrix[0]) <= 3:
            if len(self.features_matrix[0]) == 2:
                for i in range(len(self.classes_vector)):
                    self.axis.scatter(self.features_matrix[i][0], self.features_matrix[i][1], s=100,
                                      color=self.colors[self.classes_vector[i]])
            else:
                for i in range(len(self.classes_vector)):
                    self.axis.scatter(self.features_matrix[i][0], self.features_matrix[i][1], self.features_matrix[i][2],
                                      s=100, color=self.colors[self.classes_vector[i]])

            plt.show()


def eval_svm(num_of_features=4):
    if num_of_features > 4 or num_of_features < 2:
        num_of_features = 4

    dataset = load_iris()

    target = dataset.target[dataset.target != 2]
    target[target == 0] = -1

    y_test = np.append(target[:10], target[90:])
    y_train = target[10:90]

    features = dataset.data[dataset.target != 2]

    features_test = np.vstack((features[:10, :num_of_features], features[90:, :num_of_features]))
    features_train = features[10:90, :num_of_features]

    svm = SVM()

    svm.train(features_train, y_train)

    for i in range(len(y_test)):
        print('prediction: ', svm.predict(features_test[i]))
        print('class: ', y_test[i])

    svm.display()


eval_svm()
eval_svm(3)
eval_svm(2)
