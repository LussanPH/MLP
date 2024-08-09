import math 
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas

from sklearn import datasets

class Perceptron:
    def __init__(self, taxa, pesos, atributes, teta, f):
        i = 0
        self.atributes = atributes
        self.teta = teta
        self.taxa = taxa
        self.pesos = pesos
        self.f = f

    def functionG(self, u):
        r = 1/(1 + math.exp(-u)) 
        return r
    
    def ReLu(self, u):
        if(u > 0):
            return u
        else:
            return 0

    def somatório(self, ind, pesos):
        soma = np.dot(ind, pesos) + self.teta
        return soma      
    
    def rodarPercep(self):
        u = self.somatório(self.atributes, self.pesos)
        if(self.f == 1):       
            yP = self.functionG(u)
        else:
            yP = self.ReLu(u)    
        return yP



class MLP:
    def __init__(self, cOcultas, neuroniosCOcultas, neuroniosTotais, tax, base, id, nSaidas):
        self.cOc = cOcultas
        self.nCOc = neuroniosCOcultas
        self.nTotais = neuroniosTotais
        self.tax = tax
        self.id = id
        self.nSaidas = nSaidas
        self.tetas = []
        k = 0
        vez = 0
        self.data = pandas.read_csv(base)
        n = np.max(self.nCOc)
        lista = [nSaidas, n, len(self.data.iloc[0, :].values) - 2, len(self.data.iloc[0, :].values) - 1]
        m = np.max(lista)                
        self.weights = np.zeros(shape = (self.nTotais, m))  
        for _ in range(self.nTotais):
            self.tetas.append(-1)
        for _ in range(self.cOc):
            self.gerarPesos(self.nCOc[_ - vez], k, vez)
            k += self.nCOc[_] 
            if(vez == 0):
                vez+=1  
        self.gerarPesos(self.nCOc[-1], k, 1)   #TA GERANDO OS PESOS E AS BIAS, FALTA RODAR E OS SELF X E Y, QUE SÃO FORA DELE PQ ELE SO RECEBE OS PARAMETROS E CRIA UM MLP          
        print(self.weights) 
        self.tetas = np.array(self.tetas)
        self.weights = np.array(self.weights)            
        self.gerarXy()
        self.rodarMLP()
                    
    def gerarXy(self):
        self.X = self.data
        self.y = self.data.target
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def gerarPesos(self, _, k, vez):
        if(self.id == 0 and vez == 0):
            for _2 in range(_):
                for _3 in range(len(self.data.iloc[0, :].values) - 1):
                    self.weights[_2][_3] = rd.uniform(-1, 1)
        elif(self.id == 1 and vez == 0):
            for _2 in range(_):
                for _3 in range(len(self.data.iloc[0, :].values) - 2):
                    self.weights[_2][_3] = rd.uniform(-1, 1)
        else:
            for _2 in range(len(self.weights[:, 0]) - k):
                for _3 in range(_):
                    self.weights[_2 + k][_3] = rd.uniform(-1, 1)                                     
    
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
    
    def DerivadaReLu(self, u):
        saida = []
        for _ in u:
            if(_ > 0):
                saida.append(1)
            else:
                saida.append(0)
        saida = np.array(saida)        
        return saida        
    
    def calculoGradiente(self, saidas, saidas2, saidasf, weights1, weights2, weights3, soma1, soma2):
        saidasf = np.array(saidasf).reshape((3,1))
        saidas2 = np.array(saidas2).reshape((4,1))
        saidas = np.array(saidas).reshape((4,1))
        weights1 = np.array(weights1).reshape((4,4))
        weights2 = np.array(weights2).reshape((4,4))
        weights3 = np.array(weights3).reshape((3,4))
        soma1 = np.array(soma1).reshape((4,1))
        soma2 = np.array(soma2).reshape((4,1))
        self.vetor = np.array(self.vetor).reshape((3,1))
        self.atributes = np.array(self.atributes).reshape((4,1))
        gSaida = saidasf - self.vetor
        gSaidafWeights = np.dot(gSaida, np.transpose(saidas2))
        gSaidaTetas = gSaida
        for _ in range(3):
            self.weights[_+8] = self.weights[_+8] - self.tax*gSaidafWeights[_]
            self.tetas[_+8] = self.tetas[_+8] - self.tax*gSaidaTetas[_][0]
        gCamada2 = np.dot(np.transpose(weights3),gSaida) * self.DerivadaReLu(soma2).reshape((4,1))
        gCamada2Weights = np.dot(gCamada2, np.transpose(saidas))
        gCamada2Tetas = gCamada2 
        for _ in range(4):
            self.weights[_+4] = self.weights[_+4] - self.tax*gCamada2Weights[_]
            self.tetas[_+4] = self.tetas[_+4] - self.tax*gCamada2Tetas[_][0]
        gCamada1 = np.dot(np.transpose(weights2),gCamada2) * self.DerivadaReLu(soma1).reshape((4,1))
        gCamada1Weights = np.dot(gCamada1, np.transpose(self.atributes))
        gCamada1Tetas = gCamada1     
        for _ in range(4):
            self.weights[_] = self.weights[_] - self.tax*gCamada1Weights[_]
            self.tetas[_] = self.tetas[_] - self.tax*gCamada1Tetas[_][0]

    #def avaliar(self):
        #previsao = predict(self.X_teste)
        #acuracia = metrics.accuracy_score(self.y_teste, previsao)        

    def rodarMLP(self):
        for i,self.atributes in enumerate(self.X):
            saidas = []  
            saidas2 = [] 
            saidaf = []  
            weights3 = []
            weights2 = []
            weights1 = []
            soma2 = []
            soma1 = []
            for _ in range(4): 
                weights1.append(self.weights[_])
                soma1.append(Perceptron(self.tax, self.weights[_], self.atributes, self.tetas[_], 0).somatório(self.atributes, self.weights[_]))
                z = Perceptron(self.tax, self.weights[_], self.atributes, self.tetas[_], 0).rodarPercep()
                saidas.append(z)
            for _ in range(4):  
                weights2.append(self.weights[_+4])
                soma2.append(Perceptron(self.tax, self.weights[_+4], self.atributes, self.tetas[_+4], 0).somatório(self.atributes, self.weights[_+4]))
                z = Perceptron(self.tax, self.weights[_+4], saidas, self.tetas[_+4], 0).rodarPercep()
                saidas2.append(z)
            for _ in range(3): 
                weights3.append(self.weights[_+8])
                z = Perceptron(self.tax, self.weights[_+8], saidas2, self.tetas[_+8], 1).rodarPercep()        
                saidaf.append(z)
            custo = self.calculoCusto(i, saidaf)
            self.calculoGradiente(saidas, saidas2, saidaf, weights1, weights2, weights3, soma1, soma2)
            
                
             
            
MLP(2, [4, 4], 11, 0.1, "Iris.csv", 1, 3)
