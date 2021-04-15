library(ggplot2)
require(gridExtra)
rho <- function(k, alpha) alpha^k
plot1 = qplot(0:10, rho(0:10, 0.7),geom="path")
plot2 = qplot(0:10, rho(0:10, -0.7),geom="path")
grid.arrange(plot1, plot2, nrow=2)
set.seed(1)
x <- w <- rnorm(100)
for (t in 2:100) x[t] <- 0.7 * x[t - 1] + w[t]
plot(x, type = "l")
acf(x)
pacf(x)
library(forecast)
x.ar = auto.arima(x, max.q=0, stationary=TRUE, seasonal=FALSE)
www<-"https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/pounds_nz.dat"
Z <- read.table(www,header=T)
Z.auto <- auto.arima(Z)
Z2.auto=auto.arima(Z,allowmean=TRUE)
Z3.ar = auto.arima(Z,max.q=0,allowmean=TRUE,stationary=TRUE)
glob<-"https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/global.dat"
Global<-scan(glob)
plot(Global)
Global.ts=ts(Global,fr=12)
Global.ar <- ar(aggregate(Global.ts, FUN = mean), method = "mle")
mean(aggregate(Global.ts,FUN=mean))
Global.ar$order
acf(Global.ar$res[-(1:Global.ar$order)], lag = 50)
library(urca)
www2 = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/q-gdp4708.txt"
q.gdp4708 <- read.table(www, header=T)
a = ur.kpss(q.gdp4708$xrate,type = "tau")  
summary(a)
