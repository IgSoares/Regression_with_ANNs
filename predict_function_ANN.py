import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------------- #

def read_data():

	df = pd.read_excel("Dados_Trabalho_RNA.xlsx",index_col=0)
	list_values = df.values.tolist()

	x = []
	y = []

	for item in list_values: # dados de entrada e saida
		x.append([item[0]])
		y.append([item[1]])

	x = np.array(x)
	y = np.array(y)

	return x,y


def train_net(inputs,outputs):

	input_size = 1
	hidden_layers_dimensions = [10,10,10] # numero de nós em cada camada oculta (3 camadas ocultas)
	output_size = 1

	# Construção da arquitetura da rede neural (3 camadas ocultas com 10 neurônios cada)
	model = torch.nn.Sequential(torch.nn.Linear(input_size,hidden_layers_dimensions[0]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[0],hidden_layers_dimensions[1]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[1],hidden_layers_dimensions[2]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[2],output_size))

	# Definição de hiperparâmetros
	learning_rate = 1e-4
	epochs = 50000

	# Função de perda e otimizador da rede neural
	loss_fn = torch.nn.MSELoss(reduction="sum")
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for t in range(epochs):
		# Forward propagation
		y_pred = model(inputs)
		loss = loss_fn(y_pred,outputs)
		
		# Back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if t % 250 == 249:
			print(t,loss.item())

	return model

# --------------------------------------------------------------------------------------------------------------------------------------------- #

# Lendo dados do arquivo .csv
x,y = read_data()

# Separando os dados em 80% para treino, e 20% para teste 
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2, random_state=1234)

train_x = torch.from_numpy(train_x.astype(np.float32))
test_x  = torch.from_numpy(test_x.astype(np.float32))
train_y = torch.from_numpy(train_y.astype(np.float32))
test_y  = torch.from_numpy(test_y.astype(np.float32))

# Treinamento da rede
model = train_net(train_x,train_y)

# Listas dos valores de entrada e saida (valores preditos pela rede)
inputs = []
outputs = []

# Estimativas a partir da rede treinada
for i in range(len(x)):
	test_value = torch.tensor([i],dtype=torch.float32)
	inputs.append(i)
	outputs.append(model(test_value).item())

# Plot dos graficos
plt.figure()
plot_orig, = plt.plot(x,y)
plot_pred, = plt.plot(inputs,outputs,"ro")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Curvas original e valores estimados função")
plt.legend([plot_orig,plot_pred],["Curva Original","Valores preditos pela rede"])
plt.show()