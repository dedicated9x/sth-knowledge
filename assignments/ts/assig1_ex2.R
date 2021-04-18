# TODO sprawdzic te modele

www<-"https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/sp5may.dat"
Z <- read.table(www,header=T)

s = Z$lnspot
f = Z$lnfuture

y = diff(f)
x = diff(s)

lm1  <- lm(y~x)
summary(lm1)

lm2  <- lm(y~-1+x)
summary(lm2)


res = lm1$residuals
plot(res, type='l')
acf(res)
# TODO gdzie on w tej pracy wybiera miedzy modelami


#CLEAR
rm(list = ls())
dev.off()