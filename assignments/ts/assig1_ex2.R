library(nlme)
library(forecast)
www<-"https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/sp5may.dat"
Z <- read.table(www,header=T)

s = Z$lnspot
f = Z$lnfuture
y = diff(f)
x = diff(s)

lm1  <- lm(y~x)
summary(lm1)
print("Intercept is not statisticaly significant, so we won't consider it further.")
lm1  <- lm(y~-1+x)
summary(lm1)
cat("Beta1 is equal to ", lm1$coefficients['x'])

res = lm1$residuals

m1 <- res
m2 <- auto.arima(res, max.d=0)
m3 <- auto.arima(res, max.d=0, max.q=0)


print(m2)
print(m3)

par(mfrow=c(2,2))
acf(m1)
acf(m2$residuals)
acf(m3$residuals)
par(mfrow=c(1,1))

conclusion1 <- "White noise is not a good model for residuals.
ARMA(1,1) gives better model than white-noise.
Restriction to AR does not give a improvement."
cat(conclusion1)


glm1 = gls(y~x, correlation=corARMA(p=1, q=1))
conclusion2 <- "Unfortunately, GLS algorithm does not converge. BUT ...
I can tell what people usually see in such situations. They see no difference in 
coefficients and big difference in sigma squares.
"
cat(conclusion2)


#CLEAR
rm(list = ls())
dev.off()
