# there is true relationship <==> Cointegrated <==> residuals are stationary <==> .
# Engle-Granger two-step method


w.path1 <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs1yr.txt"
w.path3 <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs3yr.txt"
w.gs1yr <- read.table(w.path1, header=T)
w.gs3yr <- read.table(w.path3, header=T)

r1 <- w.gs1yr$rate
r3 <- w.gs3yr$rate
m1  <- lm(r3~r1)
summary(m1)
conclusion1 <- "Linear relationships between series is statistically significant, but at 
this moment it does not tell us anything. In order to conclude from regression,
first we need to do is to to prove that variables cointegrate."
cat(conclusion1)


# source: https://economics.stackexchange.com/questions/12497/why-does-slowly-decaying-acf-indicate-that-a-time-series-is-non-stationary

plot(m1$residuals, type='l')
acf(m1$residuals, lag=36)
conclusion2 <- "Acf plot of residuals suggests that 'r1' and 'r3' does not 
cointegrate."
cat(conclusion2)
c1 <- diff(r1)
c3 <- diff(r3) 
m2 <- lm(c3~-1+c1) # Comment: '-1' forces no intercept. 
summary(m2)
prin('nw1')


acf(m2$residuals, lag=36)
m3 <- arima(c3, order=c(0,0,1), xreg=c1, include.mean=F)
m3
prin('nw2')


rsq <- (sum(c3^2) - sum(m3$residuals^2)) / sum(c3^2)
rsq
prin('nw3')

#CLEAR
rm(list = ls())
dev.off()
