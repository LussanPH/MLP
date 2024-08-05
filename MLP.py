import math 
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
        r = 1/(1 + math.exp(-u)) 
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
        z1 = math.exp(saidas[0])/(math.exp(saidas[0]) + math.exp(saidas[1]) + math.exp(saidas[2]))
        probabilidades.append(z1)
        z2 = math.exp(saidas[1])/(math.exp(saidas[0]) + math.exp(saidas[1]) + math.exp(saidas[2]))
        probabilidades.append(z2)
        z3 = math.exp(saidas[2])/(math.exp(saidas[0]) + math.exp(saidas[1]) + math.exp(saidas[2])) 
        probabilidades.append(z3) 
        return probabilidades
    
    def calculoCusto(self, i, prob):
        yReal = self.y[i]
        self.vetor = []
        if (yReal == 0):
            self.vetor = [1,0,0]
        elif(yReal == 1):
            self.vetor = [0,1,0]
        else:
            self.vetor = [0,0,1]
        custo = -(self.vetor[0] * math.log(prob[0]) + self.vetor[1] * math.log(prob[1]) + self.vetor[2] * math.log(prob[2]))
        return custo
    
    def functionG(self, u):
        r = 1/(1 + math.exp(-u)) 
        return r
    
    def calculoGradiente(self, saidas, saidas2, saidasf, weights1, weights2, weights3, soma1, soma2, soma3):
        saidasf = np.array(saidasf).reshape((3,1))
        saidas2 = np.array(saidas2).reshape((4,1))
        saidas = np.array(saidas).reshape((4,1))
        weights1 = np.array(weights1).reshape((4,4))
        weights2 = np.array(weights2).reshape((4,4))
        weights3 = np.array(weights3).reshape((3,4))
        soma1 = np.array(soma1).reshape((4,1))
        soma1 = np.array(soma2).reshape((4,1))
        soma1 = np.array(soma3).reshape((3,1))
        self.vetor = np.array(self.vetor).reshape((3,1))
        cSaida = saidasf - self.vetor
        gSaidafWeights = np.dot(cSaida, np.transpose(saidas2))
        gSaidaTetas = cSaida
        c2 = np.dot(np.transpose(weights3),cSaida) * self.functionG(soma1)#APLICAR A FUNÇÃO ReLu para conseguir calcular a derivada


    def rodarMLP(self):
        for i,atributes in enumerate(self.X):
            saidas = []  
            saidas2 = [] 
            saidaf = []  
            weights3 = []
            weights2 = []
            weights1 = []
            soma3 = []
            soma2 = []
            soma1 = []
            for _ in range(4): 
                weights1.append(self.weights[_])
                soma1.append(Perceptron(self.tax, self.weights[_], atributes, self.tetas[_]).somatório())
                z = Perceptron(self.tax, self.weights[_], atributes, self.tetas[_]).rodarPercep()
                saidas.append(z)
            for _ in range(4):  
                weights2.append(self.weights[_+4])
                soma2.append(Perceptron(self.tax, self.weights[_], atributes, self.tetas[_]).somatório())
                z = Perceptron(self.tax, self.weights[_+4], saidas, self.tetas[_+4]).rodarPercep()
                saidas2.append(z)
            for _ in range(3): 
                weights3.append(self.weights[_+8])
                soma3.append(Perceptron(self.tax, self.weights[_], atributes, self.tetas[_]).somatório())
                z = Perceptron(self.tax, self.weights[_+8], saidas2, self.tetas[_+8]).rodarPercep()        
                saidaf.append(z)
            custo = self.calculoCusto(i, saidaf)
            self.calculoGradiente(saidas, saidas2, saidaf, weights1, weights2, weights3, soma1, soma2, soma3)
                
             
            
MLP(10, 0.1)
