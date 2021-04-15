www = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/atmospheric-carbon-dioxide-recor.csv"
carbon = read.csv(www)
carbon = carbon[-611,]
y = carbon$MaunaLoaCO2
MaunLoaCo2 = ts(data = y, frequency = 12)
?stl
output.stl = stl(MaunLoaCo2, s.window = "periodic")
plot(output.stl)
library(fpp2)
autoplot(melsyd[,"Economy.Class"]) +
        ggtitle("Economy class passengers: Melbourne-Sydney") + 
      xlab("Year") +
       ylab("Thousands")
ggseasonplot(a10, year.labels=TRUE, year.labels.left=TRUE) +
       ylab("$ million") +
       ggtitle("Seasonal plot: antidiabetic drug sales")
ggseasonplot(a10, polar=TRUE) +
      ylab("$ million") +
       ggtitle("Polar seasonal plot: antidiabetic drug sales")
ggsubseriesplot(a10) +
      ylab("$ million") +
       ggtitle("Seasonal subseries plot: antidiabetic drug sales")
autoplot(elecdemand[,c("Demand","Temperature")], facets=TRUE) +
  xlab("Year: 2014") + ylab("") +
  ggtitle("Half-hourly electricity demand: Victoria, Australia")
qplot(Temperature, Demand, data=as.data.frame(elecdemand)) +
  ylab("Demand (GW)") + xlab("Temperature (Celsius)")
autoplot(visnights[,1:5], facets=TRUE) +
  ylab("Number of visitor nights each quarter (millions)")
library(GGally)
GGally::ggpairs(as.data.frame(visnights[,1:5]))
beer2 <- window(ausbeer, start=1992)
gglagplot(beer2)
ggAcf(beer2)
aelec <- window(elec, start=1980)
autoplot(aelec) + xlab("Year") + ylab("GWh")
ggAcf(aelec, lag=48)