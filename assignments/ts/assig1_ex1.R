w.path1 <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs1yr.txt"
w.path3 <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs3yr.txt"
w.gs1yr <- read.table(w.path1, header=T)
w.gs3yr <- read.table(w.path3, header=T)

r1 <- w.gs1yr$rate
r3 <- w.gs3yr$rate
m1  <- lm(r3~r1)
summary(m1)
conclusion1 <- "One can say: linear relationships between series seems to be significant.
However, it depends on whether the asymtotic normality assumptions are met. 
We need to check residuals."
cat(conclusion1)


plot(m1$residuals, type='l')
acf(m1$residuals, lag=36)
cat(conclusion2)
c1 <- diff(r1)
c3 <- diff(r3) 
m2 <- lm(c3~-1+c1) # Comment: '-1' forces no intercept. 
summary(m2)

conclusion2 <- "Acf plot shows us, that residuals in r3~r1 are heavily correlated.
It means that mentioned assumptions has not been met.
The second thing is that linear relationship between 'c1' and 'c3' seems to be significant.
"
cat(conclusion2)


acf(m2$residuals, lag=36)
m3 <- arima(c3, order=c(0,0,1), xreg=c1, include.mean=F)
m3
conclusion3 <- "Acf plot shows us, that residuals in c3~c1 are, in general, uncorrelated.
It means that linear relationship between 'c1' and 'c3' is significant.
The second thing is that MA(1) (with regressor) seems to be equally good model 
compared to model, which does not respect time (m2).
"
cat(conclusion3)


rsq <- (sum(c3^2) - sum(m3$residuals^2)) / sum(c3^2)
rsq
conclusion4 <- "R square is high. It enables accurate predictions to be made."
cat(conclusion4)

conclusion_final <- "Since differences of 'r1' and 'r3' have significant, linear
relationship with high R square, we can use that knowledge to forecast their values
in short term. Then, we can choose more profitable product.
" 
cat(conclusion_final)


#CLEAR
rm(list = ls())
dev.off()