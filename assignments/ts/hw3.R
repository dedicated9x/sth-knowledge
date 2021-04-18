library("ggplot2")
library(forecast)
library(astsa)
require(gridExtra)


# rho <- function(k, alpha) alpha^k
# plot1 = qplot(0:10, rho(0:10, 0.7), geom='path')
# plot2 = qplot(0:10, rho(0:10, -0.7), geom='path')
# grid.arrange(plot1, plot2, nrow=2)
# 
# 
# 
# set.seed(1)
# x <-  w <- rnorm(100)
# for (t in 2:200) x[t] <- 0.7 * x[t-1] + w[t]
# plot(x, type="l")
# acf(x, na.action = na.pass, lag=50)
# pacf(x, na.action = na.pass)
# 
# x.ar = auto.arima(x, max.q=0, stationary=TRUE, seasonal=FALSE)
# x.ar
# 
# www <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/pounds_nz.dat"
# Z <- read.delim(www, header=T)
# plot(Z$xrate, type='l')    
# Z.auto = auto.arima(Z)
# Z.auto
# Z.auto$sigma2 #test statystyczny


#2
glob <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/global.dat"
Global <- scan(glob)
Global.ts = ts(Global, fr=12)
#2B 
Global.annual_mean = aggregate(Global.ts, FUN=mean)
par(mfrow=c(1, 2))
acf(Global.annual_mean, lag=10)
pacf(Global.annual_mean, lag=10)
print("Acf suggests AR. Pacf suggests AR(4)")
#2CDE
Global.ar = ar(Global.annual_mean)
Global.predicted <- predict(Global.ar, n.ahead=100)
par(mfrow=c(1, 1))
ts.plot(Global.annual_mean, Global.predicted$pred, lty=1:2)
abline(h=mean(Global.annual_mean))


#3
deciles <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/m-deciles08.txt"
da = read.table(deciles, header=T)
d1 = da[,2]
plot(d1, type='l')
acf(d1, lag=50)

da.sarma <- sarima(d1, p=1, d=0, q=0, P=1, D=0, Q=1, S=12)
print(da.sarma$ttable)

jan = rep(c(1, rep(0, 11)), 39)
m1 = lm(d1~jan)
summary(m1)
print("Low p-value means model fits well.")



#CLEAR
rm(list = ls())
dev.off()

