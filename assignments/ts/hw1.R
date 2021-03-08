############
# EXERCISE 1
############
www = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/atmospheric-carbon-dioxide-recor.csv"
carbon = read.csv(www)
carbon = carbon[-611,]
y = carbon$MaunaLoaCO2
MaunLoaCo2 = ts(data = y, frequency = 12)


plot_onestep_hw <- function(series, seasonal_) {
    hw <- HoltWinters(series,seasonal=seasonal_)
    plot(hw)
    legend("topleft",c("observed","fitted"),lty=1,col=1:2)
}

plot_predicted_hw <- function(series, seasonal_, periods) {
    hw <- HoltWinters(series,seasonal=seasonal_)
    predict <-predict(hw, n.ahead=periods)
    ts.plot(series, predict, lty=1:2)
}


plot_onestep_hw(MaunLoaCo2, "mult")
plot_predicted_hw(MaunLoaCo2, "mult", 4*12)

plot_onestep_hw(MaunLoaCo2, "add")
plot_predicted_hw(MaunLoaCo2, "add", 4*12)

# They are indistinguishable in my opinion.

# CLEAR
rm(list = ls())
dev.off()





# MaunLoaCo2.hw_mult <- HoltWinters(MaunLoaCo2,seasonal="mult")
# plot(MaunLoaCo2.hw_mult)
# MaunLoaCo2.predict_mult <-predict(MaunLoaCo2.hw_mult,n.ahead=4*12)
# ts.plot(MaunLoaCo2,MaunLoaCo2.predict_mult,lty=1:2)
# 
# MaunLoaCo2.hw_add <- HoltWinters(MaunLoaCo2,seasonal="add")
# plot(MaunLoaCo2.hw_add)
# MaunLoaCo2.predict_add <-predict(MaunLoaCo2.hw_add,n.ahead=4*12)
# ts.plot(MaunLoaCo2,MaunLoaCo2.predict_add,lty=1:2)
# 
# 
# # output.stl = stl(MaunLoaCo2, s.window = "periodic")
# # plot(output.stl)
# # a <- output.stl$time.series
# # acf(a)
# # apply(a,2,sd)
# 
# 
# 
# 
# data(AirPassengers)
# AP <- AirPassengers
# str(AP)
# ?HoltWinters 
# # TODO tu jest ten seasonal
# AP.hw <- HoltWinters(AP,seasonal="mult")
# AP.hw <- HoltWinters(AP,seasonal="add")
# plot(AP.hw)
# legend("topleft",c("observed","fitted"),lty=1,col=1:2)
# AP.predict <-predict(AP.hw,n.ahead=4*12)
# ts.plot(AP,AP.predict,lty=1:2)
# 
# 
