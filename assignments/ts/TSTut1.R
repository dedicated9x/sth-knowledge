www = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/atmospheric-carbon-dioxide-recor.csv"
carbon = read.csv(www)
carbon = carbon[-611,]
y = carbon$MaunaLoaCO2
MaunLoaCo2 = ts(data = y, frequency = 12)
output.stl = stl(MaunLoaCo2, s.window = "periodic")
plot(output.stl)
a <- output.stl$time.series
acf(a)
apply(a,2,sd)
data(AirPassengers)
AP <- AirPassengers
str(AP)
?HoltWinters 
AP.hw <- HoltWinters(AP,seasonal="mult")
plot(AP.hw)
legend("topleft",c("observed","fitted"),lty=1,col=1:2)
AP.predict <-predict(AP.hw,n.ahead=4*12)
ts.plot(AP,AP.predict,lty=1:2)


