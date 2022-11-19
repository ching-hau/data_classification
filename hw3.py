import sys
import numpy as np


def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanh(x):
    f = tanh(x)
    return 1 - f * f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)

def mse_loss(true_val, pred_val):
    return ((true_val - pred_val) ** 2).mean()

class NeuralNetWork:
    def __init__(self, epochs, learning_rate):
        self.w = [np.random.normal(0, 0.1) for i in range(15)]  
        self.b = [np.random.normal(0, 0.1) for i in range(6)]  
        self.epochs = epochs
        self.learning_rate = learning_rate

    def feedforward(self, x):
        h0 = tanh(self.w[0] * x[0] + self.w[1] * x[1] + self.b[0])
        h1 = tanh(self.w[2] * x[0] + self.w[3] * x[1] + self.b[1])
        h2 = tanh(self.w[4] * x[0] + self.w[5] * x[1] + self.b[2])
        h3 = tanh(self.w[6] * x[0] + self.w[7] * x[1] + self.b[3])
        h4 = tanh(self.w[8] * x[0] + self.w[9] * x[1] + self.b[4])
        o_out = tanh(self.w[10] * h0 + self.w[11] * h1 + + self.w[12] * h2 + self.w[13] * h3 + self.w[14] * h4 + self.b[5])
        return o_out

    def train(self, train_data, train_label):
        for epoch in range(self.epochs):
            for pos, cur_label in zip(train_data, train_label):
                cur = np.array([0, 0])
                cur[0] = pos[0]
                cur[1] = pos[1]

                sum_h0 = self.w[0] * cur[0] + self.w[1] * cur[1] + self.b[0]
                h0 = tanh(sum_h0)
                sum_h1 = self.w[2] * cur[0] + self.w[3] * cur[1] + self.b[1]
                h1 = tanh(sum_h1)
                sum_h2 = self.w[4] * cur[0] + self.w[5] * cur[1] + self.b[2]
                h2 = tanh(sum_h2)
                sum_h3 = self.w[6] * cur[0] + self.w[7] * cur[1] + self.b[3]
                h3 = tanh(sum_h3)
                sum_h4 = self.w[8] * cur[0] + self.w[9] * cur[1] + self.b[4]
                h4 = tanh(sum_h4)

                sum_o1 = self.w[10] * h0 + self.w[11] * h1 + self.w[12] * h2 + self.w[13] * h3 + self.w[14] * h4 + self.b[5]
                o1 = tanh(sum_o1)

                pred = (o1 + 1) / 2

                dl_dp = -2 * (cur_label - pred)

                dp_dh0 = self.w[10] * derivative_tanh(sum_o1)
                dp_dh1 = self.w[11] * derivative_tanh(sum_o1)
                dp_dh2 = self.w[12] * derivative_tanh(sum_o1)
                dp_dh3 = self.w[13] * derivative_tanh(sum_o1)
                dp_dh4 = self.w[14] * derivative_tanh(sum_o1)

                
                dp_dw10 = h0 * derivative_tanh(sum_o1)
                dp_dw11 = h1 * derivative_tanh(sum_o1)
                dp_dw12 = h2 * derivative_tanh(sum_o1)
                dp_dw13 = h3 * derivative_tanh(sum_o1)
                dp_dw14 = h4 * derivative_tanh(sum_o1)

                dp_db5 = derivative_tanh(sum_o1)
                

                dh0_dw0 = cur[0] * derivative_tanh(sum_h0)
                dh0_dw1 = cur[1] * derivative_tanh(sum_h0)
                dh0_db0 =  derivative_tanh(sum_h0)
                
                dh1_dw2 = cur[0] * derivative_tanh(sum_h1)
                dh1_dw3 = cur[1] * derivative_tanh(sum_h1)
                dh1_db1 =  derivative_tanh(sum_h1)

                dh2_dw4 = cur[0] * derivative_tanh(sum_h2)
                dh2_dw5 = cur[1] * derivative_tanh(sum_h2)
                dh2_db2 =  derivative_tanh(sum_h2)
                
                dh3_dw6 = cur[0] * derivative_tanh(sum_h3)
                dh3_dw7 = cur[1] * derivative_tanh(sum_h3)
                dh3_db3 =  derivative_tanh(sum_h3)
                                
                dh4_dw8 = cur[0] * derivative_tanh(sum_h4)
                dh4_dw9 = cur[1] * derivative_tanh(sum_h4)
                dh4_db4 =  derivative_tanh(sum_h4)

                self.w[0] -= self.learning_rate * dl_dp * dp_dh0 * dh0_dw0
                self.w[1] -= self.learning_rate * dl_dp * dp_dh0 * dh0_dw1
                self.w[2] -= self.learning_rate * dl_dp * dp_dh1 * dh1_dw2
                self.w[3] -= self.learning_rate * dl_dp * dp_dh1 * dh1_dw3
                self.w[4] -= self.learning_rate * dl_dp * dp_dh2 * dh2_dw4
                self.w[5] -= self.learning_rate * dl_dp * dp_dh2 * dh2_dw5
                self.w[6] -= self.learning_rate * dl_dp * dp_dh3 * dh3_dw6
                self.w[7] -= self.learning_rate * dl_dp * dp_dh3 * dh3_dw7
                self.w[8] -= self.learning_rate * dl_dp * dp_dh4 * dh4_dw8
                self.w[9] -= self.learning_rate * dl_dp * dp_dh4 * dh4_dw9
                self.w[10] -= self.learning_rate * dl_dp * dp_dw10
                self.w[11] -= self.learning_rate * dl_dp * dp_dw11
                self.w[12] -= self.learning_rate * dl_dp * dp_dw12
                self.w[13] -= self.learning_rate * dl_dp * dp_dw13
                self.w[14] -= self.learning_rate * dl_dp * dp_dw14
                self.b[0] -= self.learning_rate * dl_dp * dp_dh0 * dh0_db0
                self.b[1] -= self.learning_rate * dl_dp * dp_dh1 * dh1_db1
                self.b[2] -= self.learning_rate * dl_dp * dp_dh2 * dh2_db2
                self.b[3] -= self.learning_rate * dl_dp * dp_dh3 * dh3_db3
                self.b[4] -= self.learning_rate * dl_dp * dp_dh4 * dh4_db4
                self.b[5] -= self.learning_rate * dl_dp * dp_db5

 
        
    def predict_data(self, data):
        result = []
        for pos in data: 
            # np.append(result, self.feedforward(pos))
            cur_result = self.feedforward(pos)
            if cur_result >= 0:
                result.append(1)
            else:
                result.append(0)
        return result
                
def verify_data(pred_label, real_label):
    count = 0
    for i in range(len(pred_label)):
        # cur = str(pred_label[i]) + ' ' + str(real_label[i])
        if(pred_label[i] == real_label[i]):
            count+=1
    return count / len(pred_label)

        

# train_data_file = sys.argv[1]
# train_label_file = sys.argv[2]
# test_data_file = sys.argv[3]

train_data_file = "circle_train_data.csv"
train_label_file = "circle_train_label.csv"
test_data_file = "circle_test_data.csv"
test_label_file = "circle_test_label.csv"

# train_data_file = "xor_train_data.csv"
# train_label_file = "xor_train_label.csv"
# test_data_file = "xor_test_data.csv"
# test_label_file = "xor_test_label.csv"

# train_data_file = "spiral_train_data.csv"
# train_label_file = "spiral_train_label.csv"
# test_data_file = "spiral_test_data.csv"
# test_label_file = "spiral_test_label.csv"

# train_data_file = "gaussian_train_data.csv"
# train_label_file = "gaussian_train_label.csv"
# test_data_file = "gaussian_test_data.csv"
# test_label_file = "gaussian_test_label.csv"

train_data = np.loadtxt(train_data_file, delimiter=",")
train_label = np.loadtxt(train_label_file, delimiter=",")
test_data = np.loadtxt(test_data_file, delimiter=",")
# label_data = np.loadtxt(test_label_file, delimiter=",")


my_neural_network = NeuralNetWork(500, 0.1)
my_neural_network.train(train_data, train_label)
predict_result = my_neural_network.predict_data(test_data)
np.savetxt("test_predictions.csv", predict_result, fmt = '%i')






