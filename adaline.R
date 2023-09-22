rm(list=ls())

library('corpcor')

# Criando os dados
s1 <- 0.4 # escala que define o quão disperso os dados serão para as suas classes
s2 <- 0.4
num_amostras <- 200
xc1 <- matrix(rnorm(num_amostras*2), ncol = 2)*s1 + matrix(c(2,2), nrow = num_amostras, ncol = 2) # Estamos criando uma matriz que irá armazenar os dados centrados no ponto (2,2). Ou seja, geramos pontos aleatórios dos dados centrados no ponto (2,2).
xc2 <- matrix(rnorm(num_amostras*2), ncol = 2)*s2 + matrix(c(4,4), nrow = num_amostras, ncol = 2) # Estamos criando uma matriz que irá armazenar os dados centrados no ponto (4,4). Ou seja, geramos pontos aleatórios dos dados centrados no ponto (4,4).

plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0,8), ylim = c(0,8)) # xc1[,1] e xc2[,2] são as coordenadas x e y dos pontos a serem plotados para a classe 1 (xc1)

par(new=T) # Efeito de sobreposição no gráfico, afim de podermos adicionar mais de um gráfico na mesma "página"

plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0,8), ylim = c(0,8)) # xc2[,1] e xc2[,2] são as coordenadas x e y dos pontos a serem plotados para a classe 2 (xc2)

X = rbind(xc1, xc2) # rbind irá combinar as matrizes xc1 e xc2 com suas colunas e linhas

yc1 <- matrix(1, nrow = num_amostras, ncol = 1) # matriz Y falada no enunciado do exercício. Os dados rotulados por essa classe são representados como '1' justamente para indicar a classe positiva ou de interesse.
yc2 <- matrix(-1, nrow = num_amostras, ncol = 1) # Nesse caso, os dados rotulados por essa classe são representados como '-1' justamente para indicar uma classe negativa ou de não interesse.

Y = rbind(yc1, yc2)

# Adaline

    cbind(yc1, yc2)

W <- pseudoinverse(X) %*% Y

