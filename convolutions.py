import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

def changeImgToGray():
    image = cv2.imread('97447370.jpg')
    image = cv2.resize(image, (200, 200))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0  # Sửa lỗi chuyển đổi màu
    return img_gray

class Conv2d:
    def __init__(self, input, numofKernel=8, kernelSize=3, padding=0, stride=1):
        self.input = np.pad(input, ((padding, padding), (padding, padding)), 'constant')
        self.stride = stride
        self.kernel = np.random.randn(numofKernel, kernelSize, kernelSize)
        self.results = np.zeros((int((self.input.shape[0] - self.kernel.shape[1]) / self.stride) + 1,
                                int((self.input.shape[1] - self.kernel.shape[2]) / self.stride) + 1,
                                self.kernel.shape[0]))

    def getRoi(self):
        for row in range(int((self.input.shape[0] - self.kernel.shape[1]) / self.stride) + 1):
            for col in range(int((self.input.shape[1] - self.kernel.shape[2]) / self.stride) + 1):
                roi = self.input[row * self.stride: row * self.stride + self.kernel.shape[1],
                                col * self.stride: col * self.stride + self.kernel.shape[2]]
                yield row, col, roi

    def operator(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getRoi():
                self.results[row, col, layer] = np.sum(roi * self.kernel[layer])

        return self.results

# class Relu
class Relu:
    def __init__(self, input_data):
        self.input_data = input_data
        self.results = np.zeros((self.input_data.shape[0], self.input_data.shape[1], self.input_data.shape[2]))

    def operator(self):
        for layer in range(self.input_data.shape[2]):
            for row in range(self.input_data.shape[0]):
                for col in range(self.input_data.shape[1]):
                    self.results[row, col,layer] = 0 if self.input_data[row, col,layer] < 0 else self.input_data[row, col,layer]
        return self.results

class RekyRelu:
    def __init__(self, input_data):
        self.input_data = input_data
        self.results = np.zeros((self.input_data.shape[0], self.input_data.shape[1], self.input_data.shape[2]))

    def operator(self):
        for layer in range(self.input_data.shape[2]):
            for row in range(self.input_data.shape[0]):
                for col in range(self.input_data.shape[1]):
                    self.results[row, col,layer] = 0.1*self.input_data[row, col,layer] if self.input_data[row, col,layer] < 0 else self.input_data[row, col,layer]
        return self.results

class MaxPooling:
    def __init__(self,input,poolingSize = 2):
        self.input=input
        self.poolingSize=poolingSize
        self.results= np.zeros((int(self.input.shape[0]/self.poolingSize),
                                int(self.input.shape[1]/self.poolingSize)
                                ,self.input.shape[2]))
        

    def operator(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.results[row, col,layer] = np.max(self.input[row*self.poolingSize : row*self.poolingSize + self.poolingSize
                                                             ,col*self.poolingSize:col*self.poolingSize + self.poolingSize
                                                             ,layer])
        return self.results

class SoftMax:
    def __init__(self,input , nodes):
        self.input=input
        self.nodes=nodes
        #y = w0 +w(i) * x
        self.flatten = self.input.flatten 
        self.weigth = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0]
        self.bias = np.random.randn(nodes)
    def operator(self):
        totals = np.dot(self.flatten , self.weigth) * self.bias
        exp = np.exp(totals)
        return exp/sum(exp)


# Instantiate the Conv2d class and perform convolution
img_gray = changeImgToGray()

conv2d = Conv2d(img_gray, 8, 3, padding=2, stride=1)
img_gray_2d = conv2d.operator()
# relu
conv2d_relu = RekyRelu(img_gray_2d)
img_gray_2d_relu = conv2d_relu.operator()
#Maxpooling
img_maxpooling = MaxPooling(img_gray_2d_relu).operator()
plt.imshow(img_maxpooling[:, :, 0], cmap='gray') 
plt.show()
