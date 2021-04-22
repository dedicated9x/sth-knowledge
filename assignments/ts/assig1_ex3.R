library(forecast) 

www <- "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/q-gdpdef.txt"
Z <- read.table(www, header=T)

deflator <- Z$gdpdef
d = diff(deflator)
m1 <- auto.arima(d)
m1
m2 <- arima(d, c(0,0,0))
print(m1$sigma2)
print(m2$sigma2)
acf(m1$residuals)

conclusion1 <- "ARIMA(0,1,1) have substantially lower sigma2 compared to
white noise. In addition, ACF of its residuals shows that they do not correlate.
It proves Arima's validity.
" 
cat(conclusion1)


fcast <- forecast(m1, h=4)
deflator2009 <- tail(deflator, 1) + cumsum(fcast$mean)
deflator <- c(deflator, deflator2009)
quarterly_inflation <- diff(deflator) / head(deflator, -1)
dates <-seq(as.Date("2009-01-1"), as.Date("2010-1-1"), by="months")[seq(1, 12, 3)]
plot(dates, tail(quarterly_inflation, 4), ylim = c(0, 0.005), ylab='inflation rate', 
     xlab='2009', col='red', pch=15)

conclusion2 <- " As we can see, forecasted values of inflation are constant and
equal to approximately 0.45%
" 
cat(conclusion2)


#CLEAR
rm(list = ls())
dev.off()