import numpy as np
import random

class dataset:
    #Length of dataset
    num = 10000

    def __init__(self, num = 10000):
        self.num = num

    def get(self, add_noise=False):
        n = self.num
        #Noise is not present
        if (add_noise == False):
            arr = []
            for i in range(int(n / 2)):
                x = random.uniform(0, 6.28318530718)
                arr0 = []
                arr1 = []
                arr0.append(np.cos(x))
                arr0.append(np.sin(x))
                #Label 0
                arr0.append(0)
                arr1.append(np.cos(x))
                arr1.append(3 + np.sin(x))
                #Label 1
                arr1.append(1)
                arr.append(arr0)
                arr.append(arr1)
            arr = np.array(arr)
            #Shuffling the dataset
            np.random.shuffle(arr)
            return arr
        else:
            #Data with Gaussian Noise
            mean = 0
            std = 0.1
            array = []
            for i in range(int(n / 2)):
                #Taking angles from 0 to 2pi
                x = random.uniform(0, 6.28318530718)
                arr0 = []
                arr0.append(np.cos(x))
                arr0.append(np.sin(x))
                array.append(arr0)
            noise = np.random.normal(mean, std, [int(n / 2), 2])
            #Adding the noise
            arr_final = array + noise
            #Label 0
            arr_final = np.insert(arr_final, 2, 0, axis=1)
            array.clear()
            for i in range(int(n / 2)):
                x = random.uniform(0, 6.28318530718)
                arr0 = []
                arr0.append(np.cos(x))
                arr0.append(3 + np.sin(x))
                array.append(arr0)
            arra_final = array + noise
            #Label 1
            arra_final = np.insert(arra_final, 2, 1, axis=1)
            arra_final = np.concatenate((arr_final, arra_final), axis=0)
            np.random.shuffle(arra_final)
            return arra_final

#Sigmoid function
def sgn(n):
    if(n >= 0):
        return 1
    return -1

#Perceptron Training algorithm function
def PTA(data):
#Taking initial weights and bias to be 0
    w1 = 0
    w2 = 0
    b = 0
    while(True):
        prev_w1 = w1
        prev_w2 = w2
        prev_b = b
        for i in range(len(data)):
            x1 = data[i][0]
            x2 = data[i][1]
            m = sgn(w1*x1 + w2*x2 + b)
            curr_error = 0
            if(data[i][2] == 0):
                curr_error = -1-m
            elif(data[i][2] == 1):
                curr_error = 1-m
            if(curr_error != 0):
                w1 = w1 + curr_error*x1
                w2 = w2 + curr_error*x2
                b = b + curr_error
        #Convergence condition
        if(prev_w1 == w1 and prev_w2 == w2 and prev_b == b):
            break
    arr = []
    arr.append(w1)
    arr.append(w2)
    arr.append(b)
    return arr

#PTA function with fixed bias equal to 0
def PTA2(data):
    w1 = 0
    w2 = 0
    b = 0
    #Taking number of iterations to be 100
    for i in range(100):
        for i in range(len(data)):
            x1 = data[i][0]
            x2 = data[i][1]
            m = sgn(w1 * x1 + w2 * x2 + b)
            curr_error = 0
            if (data[i][2] == 0):
                curr_error = -1 - m
            elif (data[i][2] == 1):
                curr_error = 1 - m
            if (curr_error != 0):
                w1 = w1 + curr_error * x1
                w2 = w2 + curr_error * x2
    arr = []
    arr.append(w1)
    arr.append(w2)
    arr.append(b)
    return arr
