from math import exp
import numpy as np
import random as rd

from sklearn import datasets

class Perceptron:
    def __init__(self, taxa, pesos, atributes, teta):
        i = 0
        self.atributes = atributes
        self.teta = teta
        self.taxa = taxa
        self.pesos = pesos

    def functionG(self, u):
        r = 1/(1 + exp(-u)) 
        return r

    def somatório(self, ind, pesos):
        soma = np.dot(ind, pesos) + self.teta
        return soma      
    
    def rodarPercep(self):
        u = self.somatório(self.atributes, self.pesos)       
        yP = self.functionG(u)
        return yP



class MLP:
    def __init__(self, num, tax):
        self.num = num
        self.tax = tax
        self.tetas = []
        self.weights = np.zeros(shape = (11, 4))
        for _ in range(11):
            self.tetas.append(-1)
            if(_ < 8):      
                for _2 in range(4):
                    self.weights[_][_2] = rd.uniform(-1, 1)
            else:
                for _2 in range(3):
                    self.weights[_][_2] = rd.uniform(-1, 1)
        self.gerarXy()
        self.rodarMLP()
                    
    def gerarXy(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
    
    def softmax(self, saidas):
        probabilidades = []
        z1 = exp(saidas[0])/(exp(saidas[0]) + exp(saidas[1]) + exp(saidas[2]))
        probabilidades.append(z1)
        z2 = exp(saidas[1])/(exp(saidas[0]) + exp(saidas[1]) + exp(saidas[2]))
        probabilidades.append(z2)
        z3 = exp(saidas[2])/(exp(saidas[0]) + exp(saidas[1]) + exp(saidas[2])) 
        probabilidades.append(z3) 
        return probabilidades
        
    def rodarMLP(self):
        for atributes in self.X:
            saidas = []  
            saidas2 = [] 
            saidaf = []  
            for _ in range(4): 
                z = Perceptron(self.tax, self.weights[_], atributes, self.tetas[_]).rodarPercep()
                saidas.append(z)
            for _ in range(4):  
                z = Perceptron(self.tax, self.weights[_+4], saidas, self.tetas[_+4]).rodarPercep()
                saidas2.append(z)
            for _ in range(3): 
                z = Perceptron(self.tax, self.weights[_+8], saidas2, self.tetas[_+8]).rodarPercep()        
                saidaf.append(z)   
            print(self.softmax(saidaf))         
            
MLP(10, 0.1)
