library(ggplot2)
ar <- arima.sim(n = 300, list(ar = c(0.5,-0.5), ma = c(0.8, 0.2), sd = 1))
plot(ar)
?arima
arima(ar, order = c(2,0,2))
arima(ar, order = c(4,0,4))
library("itsmr")
hannan(ar,3,3)
library("xts")
www = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/table.csv"
table <- read.csv(www, sep=";")
# View(table)
Index <-as.Date(table$Date,"%Y-%m-%d")
price<-xts(table$Close,Index)
returns <- 100*diff(log(price),na.pad=FALSE)
arima(returns,c(1,0,0),include.mean=FALSE)
arima(returns,c(1,0,1),include.mean=FALSE)
arima(returns,c(0,0,0),include.mean=FALSE)


library("forecast")
pr<-HoltWinters(price,beta=FALSE,gamma=FALSE)
plot(pr$x,col="red",ylab="series and smoothing")
par(new=TRUE)
plot(pr$fitted[,1],col="green",xaxt="n",yaxt="n",xlab="", ylab="")

pr<-HoltWinters(price,alpha=0.1,beta=FALSE,gamma=FALSE)
plot(pr$x,col="red",ylab="series and smoothing")
par(new=TRUE)
plot(pr$fitted[,1],col="green",xaxt="n",yaxt="n",xlab="", ylab="")

pr12 <-HoltWinters(price,alpha=2/(12+1),beta=FALSE,gamma=FALSE)$fitted[,1]
pr26<-HoltWinters(price,alpha=2/(26+1),beta=FALSE,gamma=FALSE)$fitted[,1]

plot(pr$x,col="red",ylab="series and smoothing")
par(new=TRUE)
plot(pr12,col="green",xaxt="n",yaxt="n",xlab="", ylab="")
par(new=TRUE)
plot(pr26,col="yellow",xaxt="n",yaxt="n",xlab="", ylab="")
forecast<-sign(pr12-pr26)
class(forecast)
profits_price <-lag(forecast,-1)*as.ts(returns) 
plot(cumsum(profits_price),type="l")

sr <- function(X){
  if (is.null(dim(X))) X <- as.matrix(X)
  sqrt(1339) * colMeans(X,na.rm="TRUE") / sd(X,na.rm="TRUE")}
sr(profits_price)

rm(list = ls())
dev.off()
