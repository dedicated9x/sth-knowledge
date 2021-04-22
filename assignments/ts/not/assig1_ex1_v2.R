#### Downloading data:
library(forecast)

df1 <- read.table('https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs1yr.txt', header = T)
df3 <- read.table('https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/w-gs3yr.txt', header = T)

#### first part
## Performing simple linear regression on values of time series inputs 
# (not taking the order - 'time' - into consideration), as requested in the task
r1 <- df1$rate
r3 <- df3$rate

df <- cbind(df1, df3$rate)
colnames(df)[4] <- 'rate1y'
colnames(df)[5] <- 'rate3y'

m1 <- lm(df$rate3y~df$rate1y, na.action = NULL, data=df)
summary(m1)


# According to the t-test with very high cofidence (greater than 99%) we can 
# assume a linear dependency in the 1y and 3y bonds rates. Looking at the 
# standard errors we see that there is not too much variance for the 
# coefficients, whence we can expect our model to explain the  data pretty well, 
# as any deviation from the direction and position of the 'line' fitted is 
# unlikely. R-squared statistic being not that far from 1 is another 
# confirmation of the data being well explaind by our model and that assumption 
# of circa linear alignement of our data pints (B1y, B3y) was reasonable - 0.96 
# of the variance of 3Y bonds rates could be predicted from the 1Y bonds rates 
# values with our model, that is to say the line passes most of the points 
# pretty close (plotted below).
plot(r1, r3, main = "Bond rates comparison", xlab = "3 years", ylab = "1 year", pch = 19, frame = FALSE)
abline(m1, col = "red")


#### second part
# Let us take a look at the residuals
plot(m1$residuals, type='l')

# By now we have been analysing the data like it was not a time series but 
# simply data providing 1y and 3y bond rates at same times.  

# Below ACF for the residuals:
acf(m1$residuals, lag.max=36)

# In the picture above we can see that the residuals are heavily correlated. 
# Even though those are residuals of the model that does not take the time order 
# into consideration. We can treat residuals as a separate time series and in 
# this particular case it looks like it no stationary since acf does not decay 
# lets take a look at greater lag

acf(m1$residuals, lag.max=1000)

# It looks like I was right.
# We are asked to take first order diferences and analise them.

c1 <- diff(r1)
c3 <- diff(r3)
m2 <- lm(c3~-1+c1)
plot(c1,c3)
abline(m2, col='green')
summary(m2) 


# So what we see is that the first order differences are also in correlated the 
# alignment of points is linear. R-squared is not that big like in the first 
# model, but still it is large.

# What I suspect is that we had been asked to take first diferences to check if 
# there is potential trend in the model, which we could assume to be there since 
# the acf of residuals did not decay, it rather had long "ups" and "downs" 
# for lag=1000. So there might be some stochastic trend in the residual series 
# in consequence in original series as well. If I am right and there was any, it 
# should be eliminated by the differences, and this would be visible ploting acf 
# for m2$residuals, however I do not see how I should deduce that from 
# "summary(m2)" alone.


#### third part

acf(m2$residuals, lag=36)

# We see mostly we got rid of auto correlation, but once again this is the ACF 
# plot for the residuals not the original series, we may assume we got rid of 
# stochastic trends applying diffs. 

# Let us se how the ACF for c3 look like:
acf(c3, lag=100)

# It is not perfect like in theory, but for the practical purposes we can assume 
# auto correlation becomes insignificant around lag 6 and this is not far from 
# truth.

# Whence we can try to fit ARIMA, inthis particular case we are asked to fit MA, 
# which is reasonable, looking at ACF function.

m3=arima(c3, order=c(0,0,1))
summary(m3)

# I cannot say too much about this model besides the fact that  sigma-squared 
# looks pretty small even comparing to c3 values what is a good thing.
checkresiduals(m3)

#It does not look like the residuals are white noise and they shoud be.