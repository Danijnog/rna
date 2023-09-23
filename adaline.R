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
X <- cbind(X, 1) # Adiciona uma coluna de 1(uns) a matrix X

W <- pseudoinverse(X) %*% Y

# Testando dados de treinamento
# Em resumo, o código abaixo cria um ponto de teste com coordenadas (3, 3)
# e depois estende-o com um '1' no final para ser usado em um modelo de regressão linear.
yhat1 = X[1, 1] * W[1] + X[1, 2] * W[2] + X[1, 3] * W[3] # Calcula a previsão com base em uma combinação linear de X e W
yhat40 = X[40, 1] * W[1] + X[40, 2] * W[2] + X[40, 3] * W[3] # Faz o mesmo cálculo, porém para a 40 linha da matriz

yhat1 = sign(X[1, 1] * W[1] + X[1, 2] * W[2] + X[1, 3] * W[3]) # A função sign retorna 1 se o valor for positivo, -1 se for negativo e 0 se for zero.
yhat40 = sign(X[40, 1] * W[1] + X[40, 2] * W[2] + X[40, 3] * W[3])

# Cria um ponto para testar
ponto_teste = c(3, 3)

plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0, 8), ylim = c(0,8))
par(new = T)
plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
plot(ponto_teste[1], ponto_teste[2], col = 'green', pch = 2, xlim = c(0, 8), ylim = c(0, 8))

ponto_teste = c(3, 3, 1) # Adiciona um novo valor 1 ao vetor
y_teste = ponto_teste %*% W

# Agora vamos plotar pontos do grid em verde que serão usados
# para gerar a superfície de separação
plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)

seqi <- seq(0, 8, 0.5) # Gera valores de 0 a 8 incrementando de 0.5 a 0.5
seqj <- seq(0, 8, 0.5)

for(i in seqi) {
  
  for(j in seqj)
  {
    par(new = T)
    plot(i, j, col = 'green', pch = 2, xlim = c(0, 8), ylim = c(0, 8))
  }
}

# Gera o grid pra plotar a superfície de separação
seqi <- seq(0, 8, 0.5)
seqj <- seq(0, 8, 0.5)
M <- matrix(0, nrow = length(seqi), ncol = length(seqj))

ci <- 0
for(i in seqi) {
  ci <- ci + 1
  cj <- 0
  for(j in seqj)
  {
    cj <- cj + 1
    xg <- c(i, j, 1)
    
    # O comando abaixo irá classificar os pontos do grid de acordo com o comando 'sign'
    M[ci, cj] <- sign(xg %*% W)
  }
}

plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
contour(seqi, seqj, M, xlim = c(0, 8), ylim = c(0, 8), nlevels = 0) # Contour em R é usada para criar gráficos de contorno que representam visualmente os níveis de uma superfície tridimensional em um gráfico bidimensional.

# ---------------------
# Agora vamos resolver o mesmo problema anterior mas utilizando a regra Delta
# A regra delta utiliza do método do gradiente descendente com a intenção de diminuir o valor da função de erro 
# possibilitando assim a convergência para um mínimo da função de erro


eta = 0.01 # Peso de atualização de cada passo
tolerancia = 0.01 # Se o erro for menor ou igual a essa tolerancia, o treinamento é interrompido
max_epocas = 1000 # Define o número máximo de iterações permitidas para o treinamento.

N <- dim(X)[1] # dim irá atribuir o número de linhas para N
n <- dim(X)[2] # dim irá atribuir o número de colunas para n


# Runif gera um vetor de 'n' numeros aleatorios uniformemente distribuidos no intervalo [0, 1]
# o resultado é convertido em uma matriz unidimensional que terá 'n' linhas e 1 coluna. É um vetor de colunas
wt <- as.matrix(runif(n) - 0.5)

n_epocas <- 0
erro_epoca <- tolerancia + 1

erro_evec <- matrix(nrow = 1, ncol = max_epocas) # será usado para armazenas os erros ao longo das epocas

while(n_epocas < max_epocas && erro_epoca > tolerancia) {
  erro_i2 <- 0
  
  # sequencia gerada aleatoriamente para treinamento
  xseq <- sample(N)
    
  for(i in 1:N) # itera por todos os exemplos de treinamento
  {
    irand <- xseq[i] # armazena o indice de xseq a cada iteração
    yhati <- 1.0*(X[irand,] %*% wt) # yhat é a saida prevista pelo modelo linear para o treinamento atual
    erro_i <- Y[irand] - yhati # armazena o erro entre a previsão do modelo e o valor real
    gradiente <- eta * erro_i * X[irand,]
    
    # Atualização do peso w
    wt <- wt + gradiente
    
    # Erro acumulado
    erro_i2 <- erro_i2 + erro_i * erro_i
  }
  
  # Numero de epocas
  n_epocas <- n_epocas + 1
  erro_evec[n_epocas] <- erro_i2 / N # Calcula o erro medio quadratico da epoca atual e armazena na matriz erro_evec.
  
  erro_epoca = erro_evec[n_epocas] # Atribui o valor do erro medio quadratico da epoca atual a variavel erro_epoca
    
}
1. # O 1 serve para indicar que o trecho de código anterior foi encerrado
plot(erro_evec[1,], type = 'l', xlab = "Época")

# Gera o grid pra plotar a superfície de separação
seqi <- seq(0, 8, 0.5)
seqj <- seq(0, 8, 0.5)
Md <- matrix(0, nrow = length(seqi), ncol = length(seqj))

ci <- 0
for(i in seqi)
{
  ci <- ci + 1
  cj <- 0
  for(j in seqj)
  {
    cj <- cj + 1
    
    xg <- c(i, j, 1)
    
    Md[ci, cj] <- sign(xg %*% wt) # realiza o produto escalar entre o vetor 'xg' e os pesos do modelo 'wt' e, em seguida, aplica a função 'sign()' ao resultado
  }
}

plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
contour(seqi, seqj, M, col = 'green', xlim = c(0, 8), ylim = c(0, 8), nlevels = 0)
par(new = T)
contour(seqi, seqj, Md, col = 'pink', xlim = c(0, 8), ylim = c(0, 8), nlevels = 0)

# Agora resolvendo o mesmo problema por um Perceptron

step_function <- function(x) {
  ifelse(x >= 0, 1, -1)
}


eta = 0.01 # Peso de atualização de cada passo
tolerancia = 0.01 # Se o erro for menor ou igual a essa tolerancia, o treinamento é interrompido
max_epocas = 1000 # Define o número máximo de iterações permitidas para o treinamento.

N <- dim(X)[1] # dim irá atribuir o número de linhas para N
n <- dim(X)[2] # dim irá atribuir o número de colunas para n


# Runif gera um vetor de 'n' numeros aleatorios uniformemente distribuidos no intervalo [0, 1]
# o resultado é convertido em uma matriz unidimensional que terá 'n' linhas e 1 coluna. É um vetor de colunas
wt <- as.matrix(runif(n) - 0.5)

n_epocas <- 0
erro_epoca <- tolerancia + 1

erro_evec <- matrix(nrow = 1, ncol = max_epocas) # será usado para armazenas os erros ao longo das epocas

while(n_epocas < max_epocas && erro_epoca > tolerancia) {
  erro_i2 <- 0
  
  # sequencia gerada aleatoriamente para treinamento
  xseq <- sample(N)
  
  for(i in 1:N) # itera por todos os exemplos de treinamento
  {
    irand <- xseq[i] # armazena o indice de xseq a cada iteração
    yhati <- step_function((X[irand,] %*% wt)) # yhat é a saida prevista pelo modelo linear para o treinamento atual
    erro_i <- Y[irand] - yhati # armazena o erro entre a previsão do modelo e o valor real
    gradiente <- eta * erro_i * X[irand,]
    
    # Atualização do peso w
    wt <- wt + gradiente
    
    # Erro acumulado
    erro_i2 <- erro_i2 + erro_i * erro_i
  }
  
  # Numero de epocas
  n_epocas <- n_epocas + 1
  erro_evec[n_epocas] <- erro_i2 / N # Calcula o erro medio quadratico da epoca atual e armazena na matriz erro_evec.
  
  erro_epoca = erro_evec[n_epocas] # Atribui o valor do erro medio quadratico da epoca atual a variavel erro_epoca
  
}
1. # O 1 serve para indicar que o trecho de código anterior foi encerrado
plot(erro_evec[1,], type = 'l', xlab = "Época")

# Gera o grid pra plotar a superfície de separação
seqi <- seq(0, 8, 0.5)
seqj <- seq(0, 8, 0.5)
Md <- matrix(0, nrow = length(seqi), ncol = length(seqj))

ci <- 0
for(i in seqi)
{
  ci <- ci + 1
  cj <- 0
  for(j in seqj)
  {
    cj <- cj + 1
    
    xg <- c(i, j, 1)
    
    Md[ci, cj] <- sign(xg %*% wt) # realiza o produto escalar entre o vetor 'xg' e os pesos do modelo 'wt' e, em seguida, aplica a função 'sign()' ao resultado
  }
}

plot(xc1[,1], xc1[,2], col = 'red', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
plot(xc2[,1], xc2[,2], col = 'blue', xlim = c(0, 8), ylim = c(0, 8))
par(new = T)
contour(seqi, seqj, M, col = 'green', xlim = c(0, 8), ylim = c(0, 8), nlevels = 0)
par(new = T)
contour(seqi, seqj, Md, col = 'pink', xlim = c(0, 8), ylim = c(0, 8), nlevels = 0)

