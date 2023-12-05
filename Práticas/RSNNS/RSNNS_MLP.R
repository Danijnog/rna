rm(list = ls())
library("RSNNS")

N <- 200
x <- runif(N, 0, 2*pi)
y <- sin(x) + rnorm(N, 0, 0.1)

plot(x, y, col = "blue", xlim = c(0, 2*pi), ylim = c(-1, 1))

# Função de treinamento
train_mlp <- function(x, y) {
  rede <- mlp(x, y, size = 5, maxit = 2000, initFunc = "Randomize_Weights",
              initFuncParams = c(-0.3, 0.3), learnFunc = "Rprop", 
              learnFuncParams = c(0.1, 0.1), updateFunc = "Topological_Order",
              updateFuncParams = c(0), hiddenActFunc = "Act_Logistic",
              outputActFunc = "Act_Identity", # Define a função de ativação na saída da rede como linear (função identidade),
              shufflePatterns = TRUE, linOut = TRUE)
  
  return (rede)
}

# Função de previsões da rede
predict_mlp <- function(rede, x_input) {
  predict(rede, x_input)
}

rede <- train_mlp(x, y)

# Gera valores de x para a curva sen
x_vals <- seq(0, 2*pi, length.out = 500)
x_vals <- matrix(x_vals, ncol = 1)

# Depois de treinar a rede, prevê os valores de y
y_pred <- predict_mlp(rede, x_vals)

plot(x, y, col = "blue", xlim = c(0, 2*pi), ylim = c(-1, 1))
lines(x_vals, y_pred, col = "red")
legend("topright", legend = c("Dados de entrada", "Função aprox. pela MLP"), col = c("blue", "red"), lwd = 2)
