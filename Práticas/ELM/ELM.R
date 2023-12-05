# Carregue a biblioteca mlbench
library(mlbench)

# Gere o conjunto de dados
data <- mlbench.spirals(100,sd = 0.05)

# Extraia as amostras de entrada (xall) e os rótulos de saída (yall)
xall <- data$x
yall <- as.numeric(data$classes)

# Visualize o conjunto de dados
plot(data)

# Crie a matriz de pesos aleatórios Z para a camada oculta
p <- 60
Z <- replicate(p, runif(3, -0.5, 0.5))

# Normalize os dados de entrada (xall)
xall <- scale(xall)

# Adicione um termo de polarização (bias) à matriz xall normalizada
Xaug <- cbind(replicate(dim(xall)[1], 1), xall)

# Calcule a saída da camada oculta aplicando a função tangente hiperbólica
H <- as.matrix(tanh(Xaug %*% Z))

# Calcule os pesos da camada de saída da ELM
W <- pseudoinverse(H) %*% yall

# Visualize o separador linear
seqi <- seq(0, 6, 0.05)
seqj <- seq(0, 6, 0.05)
M <- matrix(0, nrow = length(seqi), ncol = length(seqj))

ci <- 0
for (i in seqi) {
  ci <- ci + 1
  cj <- 0
  for (j in seqj) {
    cj <- cj + 1
    xg <- c(1, i, j)
    Hg <- as.matrix(tanh(xg %*% Z))
    M[ci, cj] <- sign(Hg %*% W)
  }
  
  plot(data)
  par(new = T)
  # Desenhe o separador linear
  contour(seqi, seqj, M, xlim = c(0, 6), ylim = c(0, 6), nlevels = 0)
  
  legend('top', legend = ('Superfície de separação com p = 30'))
}