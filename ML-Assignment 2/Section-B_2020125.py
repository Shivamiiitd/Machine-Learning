import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import utils

data = utils.dataset(10000)
#Data with noise
with_noise = data.get(True)
#Data without noise
without_noise = data.get(False)
#Plotting both circles with noise in data
sns.set(rc={'figure.figsize':(5,10)})
sns.scatterplot(data = with_noise, hue = with_noise[:,2], x = with_noise[:, 0], y = with_noise[:, 1])
plt.show()

#Plotting both circles without noise in data
sns.scatterplot(data = without_noise, hue = without_noise[:,2], x = without_noise[:, 0], y = without_noise[:, 1])
plt.show()

#PTA on data without noise
array = utils.PTA(without_noise)
xs = np.linspace(-2, 2)
a = -array[0]/array[1]
#Decision boundary Equation
y2 = a*xs - array[2]/array[1]
#Plotting
plt.plot(xs, y2, "k-")
sns.set(rc={'figure.figsize':(10,10)})
sns.scatterplot(data = without_noise, hue = without_noise[:,2], x = without_noise[:, 0], y = without_noise[:, 1])
plt.show()

#PTA on data with noise
array = utils.PTA(with_noise)
xs = np.linspace(-2, 2)
a = -array[0]/array[1]
#Decision boundary equation
y2 = a*xs - array[2]/array[1]
#Plotting
plt.plot(xs, y2, "k-")
sns.set(rc={'figure.figsize':(10,10)})
sns.scatterplot(data = with_noise, hue = with_noise[:,2], x = with_noise[:, 0], y = with_noise[:, 1])
plt.show()

#Fixed bias PTA on data without noise
array = utils.PTA2(without_noise)
xs = np.linspace(-2, 2)
a = -array[0]/array[1]
#Decision boundary equation
y2 = a*xs - array[2]/array[1]
#Plotting
plt.plot(xs, y2, "k-")
sns.set(rc={'figure.figsize':(10,10)})
sns.scatterplot(data = data, hue = without_noise[:,2], x = without_noise[:, 0], y = without_noise[:, 1])
plt.show()

#XOR data
data_XOR = []
data_XOR.append([0, 0, 0])
data_XOR.append([0, 1, 1])
data_XOR.append([1, 0, 1])
data_XOR.append([1, 1, 0])
data_XOR = np.array(data_XOR)
#Learnable bias
array = utils.PTA(data_XOR)
xs = np.linspace(-2, 2)
if(array[1] != 0):
    a = -array[0]/array[1]
    y2 = a*xs - array[2]/array[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_XOR, hue = data_XOR[:,2], x = data_XOR[:, 0], y = data_XOR[:, 1])
    plt.show()
else:
    val = 0
    if(array[0] != 0):
        val = -array[2]/array[0]
        #Plotting
        plt.axvline(x = val)
        sns.set(rc={'figure.figsize':(10,10)})
        sns.scatterplot(data = data_XOR, hue = data_XOR[:,2], x = data_XOR[:, 0], y = data_XOR[:, 1])
        plt.show()

#Fixed bias 0
array1 = utils.PTA2(data_XOR)
xs = np.linspace(-2, 2)
if(array1[1] != 0):
    a = -array1[0]/array1[1]
    y2 = a*xs - array1[2]/array1[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_XOR, hue = data_XOR[:,2], x = data_XOR[:, 0], y = data_XOR[:, 1])
    plt.show()
else:
    #Plotting
    plt.axvline(x = 0)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_XOR, hue = data_XOR[:,2], x = data_XOR[:, 0], y = data_XOR[:, 1])
    plt.show()

#AND data
data_AND = []
data_AND.append([0, 0, 0])
data_AND.append([0, 1, 0])
data_AND.append([1, 0, 0])
data_AND.append([1, 1, 1])
data_AND = np.array(data_AND)
#Learnable bias
array = utils.PTA(data_AND)
xs = np.linspace(-2, 2)
if(array[1] != 0):
    a = -array[0]/array[1]
    y2 = a*xs - array[2]/array[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_AND, hue = data_AND[:,2], x = data_AND[:, 0], y = data_AND[:, 1])
    plt.show()
else:
    val = 0
    if(array[0] != 0):
        val = -array[2]/array[0]
        #Plotting
        plt.axvline(x = val)
        sns.set(rc={'figure.figsize':(10,10)})
        sns.scatterplot(data = data_AND, hue = data_AND[:,2], x = data_AND[:, 0], y = data_AND[:, 1])
        plt.show()

#Fixed bias 0
array1 = utils.PTA2(data_AND)
xs = np.linspace(-2, 2)
if(array1[1] != 0):
    a = -array1[0]/array1[1]
    y2 = a*xs - array1[2]/array1[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_AND, hue = data_AND[:,2], x = data_AND[:, 0], y = data_AND[:, 1])
    plt.show()
else:
    #Plotting
    plt.axvline(x = 0)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_AND, hue = data_AND[:,2], x = data_AND[:, 0], y = data_AND[:, 1])
    plt.show()

#OR data
data_OR = []
data_OR.append([0, 0, 0])
data_OR.append([0, 1, 1])
data_OR.append([1, 0, 1])
data_OR.append([1, 1, 1])
data_OR = np.array(data_OR)
#Learnable bias
array = utils.PTA(data_OR)
xs = np.linspace(-2, 2)
if(array[1] != 0):
    a = -array[0]/array[1]
    y2 = a*xs - array[2]/array[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_OR, hue = data_OR[:,2], x = data_OR[:, 0], y = data_OR[:, 1])
    plt.show()
else:
    val = 0
    if(array[0] != 0):
        val = -array[2]/array[0]
        #Plotting
        plt.axvline(x = val)
        sns.set(rc={'figure.figsize':(10,10)})
        sns.scatterplot(data = data_OR, hue = data_OR[:,2], x = data_OR[:, 0], y = data_OR[:, 1])
        plt.show()

#Fixed bias 0
array1 = utils.PTA2(data_OR)
xs = np.linspace(-2, 2)
if(array1[1] != 0):
    a = -array1[0]/array[1]
    y2 = a*xs - array1[2]/array1[1]
    #Plotting
    plt.plot(xs, y2, "k-")
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_OR, hue = data_OR[:,2], x = data_OR[:, 0], y = data_OR[:, 1])
    plt.show()
else:
    #Plotting
    plt.axvline(x = 0)
    sns.set(rc={'figure.figsize':(10,10)})
    sns.scatterplot(data = data_OR, hue = data_OR[:,2], x = data_OR[:, 0], y = data_OR[:, 1])
    plt.show()