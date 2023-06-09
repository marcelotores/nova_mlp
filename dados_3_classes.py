from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import ut
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#from corri import MLP
from mlp_amostra_3 import MLP


## Importandno dados (numpy) por padrão. Para dataframe, use (data_frame=True) com segundo parâmetro
dataSet = ut.im_data(4)


##################### TESTE

# 2 Classes
# classe1 = dataSet[:82, :]
# classe2 = dataSet[315:, :]
# classe1_classe2 = np.concatenate((classe1, classe2), axis=0)
# X = classe1_classe2[:, :24]
# y = classe1_classe2[:, 24].reshape(classe1_classe2.shape[0], 1)

# 3 Classes
dataSet_3_classes = dataSet[:315, :]
X = dataSet_3_classes[:, :24]
y = dataSet_3_classes[:, 24].reshape(dataSet_3_classes.shape[0], 1)

# 4 Classes
#X = dataSet[:, :24]
#y = dataSet[:, 24].reshape(dataSet.shape[0], 1)

# 10 Classes
# dataSet_10 = ut.im_data(10)
# X = dataSet_10[:, :24]
# y = dataSet_10[:, 24].reshape(dataSet_10.shape[0], 1)

# Pré-processamento dos dados
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Normalizar os dados
X_train /= np.max(X_train, axis=0)
X_test /= np.max(X_test, axis=0)

input_size = X_train.shape[1]
hidden_size = 3
# 3
# 0.1
# teste 30%
output_size = y_train.shape[1]


mlp = MLP(input_size, hidden_size, output_size)

peso1 = mlp.train(X_train, y_train, X_test, y_test, learning_rate=0.1, num_epochs=1000)

## P e p -> d
##
##exit()
##################
# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Converter as previsões de volta para as classes originais
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calcular a matriz de confusão
confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

# Exibir a matriz de confusão
print(confusion_mat)

confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

# Definir rótulos das classes
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Exibir a matriz de confusão
fig, ax = plt.subplots()
im = ax.imshow(confusion_mat, cmap='Blues')

# Adicionar barra de cores
cbar = ax.figure.colorbar(im, ax=ax)

# Configurar os ticks e rótulos do eixo x e y
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

# Rotacionar os rótulos do eixo x
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop para exibir os valores na matriz de confusão
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, confusion_mat[i, j], ha="center", va="center", color="black")

# Configurar título e rótulos dos eixos
ax.set_title("Matriz de Confusão")
ax.set_xlabel("Valores Preditos")
ax.set_ylabel("Valores Reais")

# Exibir a figura
plt.show()
##################
# Calcular a acurácia
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(f"Acurácia: {accuracy}")



# Plotar o gráfico de erro durante o treinamento e teste
plt.plot(range(len(mlp.train_errors)), mlp.train_errors, label='Treinamento')
plt.plot(range(len(mlp.test_errors)), mlp.test_errors, label='Teste')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.title('Gráfico de Erro durante o Treinamento')
plt.legend()
plt.show()

#exit()

############################
## Divindo os dados em treino e teste
## Retorna (numpy) ou (dataframe). Dependendo dos dados passados
#treino, teste = ut.divide_dados_treino_teste(dataSet_3_classes, 0.7)



## separando rótulos do dataset para 3 classes
#X_treino = treino[:, :24]
#y_treino = treino[:, 24].reshape(treino.shape[0], 1)
#X_teste = teste[:, :24]
#y_teste = teste[:, 24].reshape(teste.shape[0], 1)

#print(X_treino.shape)
#print(y_treino.shape)
#print(X_teste.shape)
#print(y_teste.shape)

# Pré-processamento dos dados
#encoder = OneHotEncoder(sparse=False)
#y_teste = encoder.fit_transform(y_teste)
#print(y_teste)


## separando rótulos do dataset para 4 classes
#X_treino = treino[:, :24]
#y_treino = treino[:, 24].reshape(300, 1)
# X_teste = teste[:, :24]
# y_teste = teste[:, 24].reshape(75, 1)

# y_teste = ut.converte_rotulo_3(y_teste)
# y_treino = ut.converte_rotulo_3(y_treino)

#print(ut.numero_atributo_por_classe())
#print(y_treino)
#print(y_teste)

# ## Parâmetros da Rede
# taxa_aprendizado = 0.1
# epocas = 1
# qtd_neuronios_camada_oculta = 2
# qtd_neuronios_camada_saida = 1
#


# Setando os pesos Iniciais
#qtd_col_dataset = treino2[:, :24].shape[1]
#pesos_camada_1 = np.random.uniform(-0.5, 0.5, size=(qtd_col_dataset + 1,  qtd_neuronios_camada_oculta))
#pesos_camada_2 = np.random.uniform(-0.5, 0.5, size=(qtd_neuronios_camada_oculta + 1, qtd_neuronios_camada_saida))

#mlp = Mlp(treino2[:, :24], taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida, pesos_camada_1, pesos_camada_2)

## Treino
#errors, param = mlp.treino(treino2[:, :24], treino2[:, 24].reshape(treino2.shape[0], 1))
#mlp.treino(X, y)

#mlp = Mlp(treino2[:, :24], taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

#X_treino[:10,:]
#y_treino[:10,:]
## Treino
#errors, param = mlp.treino(treino2[:, :24], treino2[:, 24].reshape(treino2.shape[0], 1))
#print(f'pesos camada oculta: {param["pesos_camada_oculta"]}')
#print(f'pesos camada saída: {param["pesos_camada_saida"]}')

#y_predicao = mlp.predicao(teste2[:, :24], param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#print(y_predicao)
## Cálculo de acurácia
#num_predicoes_corretas = (y_predicao == y_teste).sum()

#acuracia = (num_predicoes_corretas / y_teste.shape[0]) * 100
#print('Acurácia: %.2f%%' % acuracia)

#print(errors)

# Gráfico de Erro
#uti.grafico_erro(errors)