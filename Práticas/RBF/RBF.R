rm(list=ls())
library('corpcor')

# --------------- Gauss function
gauss <- function(x, m, s) {
  return(exp(-0.5 * (rowSums((x - m)^2)/(s^2))))
}

# Geração dos dados
set.seed(1000)
s1 <- 0.4
s2 <- 0.4
s3 <- 0.4
s4 <- 0.4
nc <- 100

xc1 <- matrix(rnorm(nc * 2), ncol = 2) * s1 + t(matrix(c(4, 4), ncol = nc, nrow = 2))
xc2 <- matrix(rnorm(nc * 2), ncol = 2) * s2 + t(matrix(c(4, 8), ncol = nc, nrow = 2))
xc3 <- matrix(rnorm(nc * 2), ncol = 2) * s3 + t(matrix(c(8, 4), ncol = nc, nrow = 2))
xc4 <- matrix(rnorm(nc * 2), ncol = 2) * s4 + t(matrix(c(8, 8), ncol = nc, nrow = 2))

x <- rbind(xc1, xc2, xc3, xc4)
d1 <- dim(xc1)[1]
d2 <- dim(xc2)[1]
d3 <- dim(xc3)[1]
d4 <- dim(xc4)[1]

y <- c(rep(1, d1), rep(-1, d2), rep(-1, d3), rep(1, d4))

plot(x[which(y == 1), 1], x[which(y == 1), 2], col = 'green', xlim = c(0,9), ylim = c(0,9), xlab='x1',ylab='x2')

par(new=T)

plot(x[which(y == -1), 1], x[which(y == -1), 2], col = 'red', xlim = c(0,9), ylim = c(0,9), xlab='',ylab='')

# ------------------

# 4 Gaussianas
S = 1.7 # raio da função de base gaussiana
# Valores menores de S são melhores utilizados para dados densamente agrupados.
# Valores maiores de S são melhores utilizados para dados mais dispersos entre si.
c1 = c(4, 4)
c2 = c(4, 8)
c3 = c(8, 4)
c4 = c(8, 8)

# Camada oculta que é calculada de acordo com a função gaussiana
h1 <- gauss(x, c1, S)
h2 <- gauss(x, c2, S)
h3 <- gauss(x, c3, S)
h4 <- gauss(x, c4, S)

H <- cbind(h1, h2, h3, h4, 1)
W <- pseudoinverse(H) %*% y # É calculado os pesos da rede utilizando a pseudoinversa de forma que minimize o erro entre a saída esperada da rede (y) e as saídas obtidas (yhat)

yhat <- H %*% W # Depois de calcular os pesos 'W', a rede é usada para fazer novas previsões (yhat) para novos dados de entrada (H)

# Plotando a superfície de separação

xgrid1 <- seq(1, 8, 0.2)
xgrid2 <- seq(1, 8, 0.2)

YHATG <- matrix(0, length(xgrid1), length(xgrid2))

i = 1
j = 1
k = 1

for(i in 1:length(xgrid1)) {
  
  for(j in 1:length(xgrid2)) {
    
    h1 <- gauss(t(as.matrix(c(xgrid1[i], xgrid2[j]))), c1, S)
    
    h2 <- gauss(t(as.matrix(c(xgrid1[i], xgrid2[j]))), c2, S)
    
    h3 <- gauss(t(as.matrix(c(xgrid1[i], xgrid2[j]))), c3, S)
    
    h4 <- gauss(t(as.matrix(c(xgrid1[i], xgrid2[j]))), c4, S)
    
    YHATG[i, j]=sign(t(as.matrix(c(h1, h2, h3, h4, 1))) %*% W)
    
    k = k + 1
  }
}

plot(x[which(y == 1), 1], x[which(y == 1), 2], col = 'green', xlim = c(0, 12), ylim = c(0, 12), xlab = 'x1', ylab = 'x2')

par(new = T)

plot(x[which(y == -1), 1], x[which(y == -1), 2], col = 'red', xlim = c(0, 12), ylim = c(0, 12), xlab='',ylab='')

par(new = T)

contour(xgrid1, xgrid2, YHATG, levels = 0, labels = "", xlim = c(0, 12), ylim = c(0, 12))

# -------------------

# Número de dobras (k-fold)
# K-fold é uma técnica de desempenho utilizada para algoritmos de machine learning.
# K se refere ao número de grupos que o conjunto de dados é dividido com o objetivo de treinamento e teste.
k <- 10

# Tamanho do conjunto de teste (10% dos dados)
test_size <- round(0.1 * nrow(x))

# Inicialize um vetor para armazenar as acurácias de cada dobra
accuracies <- numeric(k)

# Realize a validação cruzada
for (fold in 1:k) {
  # Divida os dados em conjuntos de treinamento e teste
  set.seed(fold)  # Defina uma semente para garantir a reprodutibilidade
  test_indices <- sample(1:nrow(x), test_size)
  train_indices <- setdiff(1:nrow(x), test_indices)
  
  x_train <- x[train_indices, ]
  y_train <- y[train_indices]
  x_test <- x[test_indices, ]
  y_test <- y[test_indices]
  
  # Treine a rede RBF nos dados de treinamento e calcule os pesos W
  S = 1.7
  c1 = c(4, 4)
  c2 = c(4, 8)
  c3 = c(8, 4)
  c4 = c(8, 8)
  
  h1 <- gauss(x_train, c1, S)
  h2 <- gauss(x_train, c2, S)
  h3 <- gauss(x_train, c3, S)
  h4 <- gauss(x_train, c4, S)
  
  H <- cbind(h1, h2, h3, h4, 1)
  W <- pseudoinverse(H) %*% y_train
  
  # Faça previsões nos dados de teste
  h1_test <- gauss(x_test, c1, S)
  h2_test <- gauss(x_test, c2, S)
  h3_test <- gauss(x_test, c3, S)
  h4_test <- gauss(x_test, c4, S)
  
  H_test <- cbind(h1_test, h2_test, h3_test, h4_test, 1) # matriz de ativação para a rede, usando as funções de base Gaussiana
  yhat_test <- sign(H_test %*% W) # essa variável que representa as previsões (saída) feitas pela RBF nos dados de teste.
  
  # Calcule a acurácia para esta dobra e armazene no vetor accuracies
  # A acurácia é calculada comparando as previsões feitas pela yhat_test 
  # com os verdadeiros dados y_test p/ determinar quantas previões estão corretas.
  
  # A acurácia é a proporção de previsões corretas em relação ao total de exemplos de teste.
  accuracy <- sum(yhat_test == y_test) / length(y_test)
  accuracies[fold] <- accuracy
}

# Calcule a média e o desvio padrão das acurácias
mean_accuracy <- mean(accuracies)
std_accuracy <- sd(accuracies)

# Imprima a acurácia média e o desvio padrão
cat("Acurácia Média:", mean_accuracy, "\n")
cat("Desvio Padrão da Acurácia:", std_accuracy, "\n")


