# Carregar bibliotecas necessárias
library(ggplot2)

# Definir a função geradora
fg <- function(x) 0.5 * x^2 + 3 * x + 10

# Definir parâmetros do ruído gaussiano
mean_noise <- 0
sd_noise <- 4

# Definir intervalo e número de amostras
x_values <- seq(-15, 10, length.out = 100)

# Definir graus dos polinômios a serem ajustados
poly_degrees <- 1:8

# Gerar os dados
set.seed(123)  # Para reprodutibilidade dos resultados
data <- data.frame(x = x_values, y = fg(x_values) + rnorm(length(x_values), mean = mean_noise, sd = sd_noise))

# Função para ajustar um polinômio e gerar um gráfico
plot_polynomial_approximation <- function(degree) {
  p <- lm(y ~ poly(x, degree), data = data)
  x_range <- seq(-15, 10, length.out = 100)
  y_pred <- predict(p, newdata = data.frame(x = x_range))
  
  p_plot <- ggplot(data, aes(x = x, y = y)) +
    geom_point() +
    geom_line(data = data.frame(x = x_range, y = y_pred), aes(x = x, y = y_pred), color = "blue") +
    geom_line(data = data.frame(x = x_range, y = fg(x_range)), aes(x = x, y = fg(x_range)), color = "red", linetype = "dashed") +
    labs(title = paste("Polynomial Degree =", degree)) +
    theme_minimal()
  
  return(p_plot)
}

# Criar e imprimir os gráficos para diferentes graus de polinômio
for (degree in poly_degrees) {
  p <- plot_polynomial_approximation(degree)
  print(p)
}
