rm(list = ls())
library(ggplot2)

# Vamos usar a função de ativação tangente hiperbólica

# Derivada da função tanh para o cálculo do gradiente
tanh_derivative <- function(x) {
  return (((2/exp(x) + exp(-x))) * (2/(exp(x) + exp(-x))))
}

# Função para treinar a MLP
train_mlp <- function(input, output, hidden_neurons, epochs, learning_rate) {
  # Inicialização dos pesos aleatórios para camada oculta e de saída
  
  input_neurons <- ncol(input)
  output_neurons <- ncol(output)
  
  set.seed(123)
  hidden_weights <- matrix(runif(input_neurons * hidden_neurons, -1, 1), nrow = input_neurons)
  output_weights <- matrix(runif(hidden_neurons * output_neurons, -1, 1), nrow = hidden_neurons)
  
  for(epoch in 1:epochs) {
    # Propagação direta (forward propagation)
    hidden_layer_input <- tanh(input %*% hidden_weights)
    output_layer <- tanh(hidden_layer_input %*% output_weights)
    
    # Calcula o erro
    output_error <- output - output_layer
    
    # Backpropagation e atualização dos pesos
    output_delta <- output_error * tanh_derivative(output_layer)
    hidden_error <- output_delta %*% t(output_weights)
    hidden_delta <- hidden_error * tanh_derivative(hidden_layer_input)
    
    output_weights <- output_weights + t(hidden_layer_input) %*% output_delta * learning_rate
    hidden_weights <- hidden_weights + t(input) %*% hidden_delta * learning_rate
  }
  
  return(list(hidden_weights = hidden_weights, output_weights = output_weights))
}

# Função p/ prever usando a MLP treinada
predict_mlp <- function(input, weights) {
  hidden_layer_input <- tanh(input %*% weights$hidden_weights)
  predicted_output <- tanh(hidden_layer_input %*% weights$output_weights)
  
  return (predicted_output)
}

# Dados de entrada e saída XOR
input_data <- matrix(c(0, 0,
                     0, 1,
                     1, 0,
                     1, 1), ncol = 2, byrow = TRUE)

output_data <- matrix(c(0, 1, 1, 0), ncol = 1)

# Parâmetros da MLP
hidden_neurons <- 3
epochs <- 10000
learning_rate <- 0.1

# Treinamento da MLP
mlp_weights <- train_mlp(input_data, output_data, hidden_neurons, epochs, learning_rate)

# Previsão usando a mlp treinada
predicted_output <- predict_mlp(input_data, mlp_weights)

# Saída espera e saída prevista
print("Saída esperada:")
print(output_data)
print("Saída prevista pela rede MLP:")
print(predicted_output)

# Gráficos
plot(input_data, col = ifelse(output_data == 1, "blue", "red"), main = "XOR Dataset", pch = 19)
legend("center", legend = c("Saída 0", "Saída 1"), col = c("red", "blue"), pch = 19)


grid <- expand.grid(X1 = seq(0, 1, length.out = 100), X2 = seq(0, 1, length.out = 100))
grid$Prediction <- predict_mlp(as.matrix(grid), mlp_weights)

ggplot(grid, aes(x = X1, y = X2, fill = Prediction)) +
  geom_tile() +
  scale_fill_gradient(low = "red", high = "blue") +
  labs(title = "Superfície de Separação") +
  theme_minimal()
  

# A partir daqui, o código é adaptado para o problema
# do Câncer de mama, carregando o pacote do mlbench.

# Pega os dados da package mlbench
library(mlbench)
library(caret)

# Realiza o tratamento dos dados para eliminação dos
# dados faltantes. Esses dados faltantes, que serão eliminados,
# são representados pela string NA.
data("BreastCancer")
data2 <- BreastCancer
data2 <- data2[complete.cases(data2),]

# Verifica as linhas dos dados pra entender a estrutura
head(data2)

# Rotula as amostras das Classes com o valor de 0 (maligno) e 1 (benigno).
# o código verifica se na coluna "Class" do dataset "data2" tem 
# os valores "benign". Se tiver, retorna TRUE e converte pra 1
# com o comando 'as.numeric'.
# Se não, retorna falso e converte pra 0 (maligno).
data2$Class <- as.numeric(data2$Class == "benign")

# Divide os dados em conjunto de treinamento e teste. 
# 70% treino, 30% teste.
set.seed(123)
indices_treino <- createDataPartition(data2$Class, p = 0.7, list = FALSE)
dados_treino <- data2[indices_treino, ]
dados_teste <- data2[-indices_treino, ]

# Lista para armazenar as acurácias de cada iteração
accuracy_values <- vector("numeric", length = 10)

for(i in 1:10) {
  set.seed(i * 123) # Para reprodutibilidade em cada iteração
  
  # 10 folds
  folds <- createFolds(data2$Class, k = 10)
  
  # Inicializar variável p/ armazenar as acurácias de cada fold
  fold_accuracies <- vector("numeric", length = 10)
  
  for(j in 1:10)
  {
    # Separar os dados em conjutno de treino e teste p/ fold j
    dados_treino <- data2[-folds[[j]], ]
    dados_teste <- data2[folds[[j]], ]
    
    # Etapas de treinamento e teste do modelo
    
    X_treino <- as.matrix(dados_treino[, -1]) # Todas as colunas exceto a primeira (classe)
    y_treino <- as.matrix(dados_treino$Class) # Coluna classe
    
    X_teste <- as.matrix(dados_teste[, -1])
    y_teste <- as.matrix(dados_teste$Class)
    
    # Converter as colunas para tipo numérico
    X_treino <- apply(X_treino, 2, as.numeric) # O parâmetro 2 indica que a função apply será aplicada a cada coluna
    # O parâmetro 'as.numerica' indica a função que será aplicada a cada coluna
    X_teste <- apply(X_teste, 2, as.numeric)
    
    # Parâmetros da MLP
    hidden_neurons <- 3
    epochs <- 500
    learning_rate <- 0.1
    
    # Vamos chamar a função de treinamento que já fizemos
    mlp_weights <- train_mlp(X_treino, y_treino, hidden_neurons, epochs, learning_rate)
    
    # Vamos chamar a função de predição que já fizemos  
    predicted_output <- predict_mlp(X_teste, mlp_weights)
    
    # Converter as saídas previstas para classe (0 ou 1) com limite de 0.5
    predicted_classes <- ifelse(predicted_output >= 0.5, 1, 0)
    
    # Avaliar a acurácia do modelo nos dados de teste
    accuracy <- mean(predicted_classes == y_teste)
    
    # Amarzenar a acurácia do fold j
    fold_accuracies[j] <- accuracy
  }
  
  # Calcular a média da acurácia para os 10 folds
  accuracy_values[i] <- mean(fold_accuracies)
}

# Média e desvio padrão das acurácias
mean_accuracy <- mean(accuracy_values)
sd_mean_accuracy <- sd(accuracy_values)

error <- (1 - mean_accuracy) * 100 
print(paste(error))

# Resultados
print(paste("Acurácia média:", mean_accuracy))
print(paste("Desvio padrão da acurácia média:", sd_mean_accuracy))
