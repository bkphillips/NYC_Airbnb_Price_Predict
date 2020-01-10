#Author: Bryan Phillips
# Description: Predicting the price of Airbnb's in NYC using matrix factorization
# Many portions of this code are influenced by the code given by the matrix factorization and regularlization section of the machine learning course

#Loading all required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(tidyr)
library(ggplot2)
library(latexpdf)
library(gridExtra)


#Dowloading csv of NYC Airbnb open data
url <- "https://raw.githubusercontent.com/bkphillips/NYC_Airbnb_Price_Predict/master/AB_NYC_2019.csv"
nyc<-read.csv(url)

#Look at nyc data summary
head(nyc)
describe(nyc)
summary(nyc)


#Looking at normal distribution of price
ggplot(nyc, aes(price)) +
  geom_histogram(bins = 30, aes(y = ..density..)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = mean(nyc$price)) 

#Looking at log distribution
ggplot(nyc, aes(price)) +
  geom_histogram(bins = 30, aes(y = ..density..)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = mean(nyc$price)) +
  scale_x_log10() 


#average price is $152
mean(nyc$price)

#Looking at number of rooms by neihborhood by type
nyc %>% group_by(neighbourhood_group, room_type) %>% summarize(n = n()) %>% 
  ggplot(aes(reorder(neighbourhood_group,desc(n)), n, fill = room_type)) + 
  xlab("Neighborhood") +
  ylab("Number of Rooms") +
  geom_bar(stat = "identity")

#Average price by neighborhood group
nyc %>% group_by(neighbourhood_group) %>%
  summarize(n = n(), avg = mean(price), se = sd(price)/sqrt(n())) %>%
  mutate(neighbourhood_group = reorder(neighbourhood_group, avg)) %>%
  ggplot(aes(x = neighbourhood_group, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#There are 221 distrinct neighborhoods
n_distinct(nyc$neighbourhood)

nyc %>% group_by(neighbourhood) %>%
  summarize(n = n(), avg = mean(price), se = sd(price)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(neighbourhood = reorder(neighbourhood, avg)) %>%
  ggplot(aes(x = neighbourhood, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


nyc %>% group_by(room_type) %>%
  summarize(n = n(), avg = mean(price), se = sd(price)/sqrt(n())) %>%
  mutate(room_type = reorder(room_type, avg)) %>%
  ggplot(aes(x = room_type, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Looking at price distribution for outliers and see that under 500 makes sense
filter_price <-nyc%>%filter( price<1000)
histogram(filter_price$price)
#Filtering everything above 0 and below 500. Removing outliers
filter_nyc<-nyc%>% filter(price>0 & price <500)
#This filter goes from 48,586 observations to 34,024 observations
y<-filter_nyc$price
set.seed(2007)
#creat 70% partition for training
train_index <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)
train_set <- filter_nyc[train_index, ]
test_set <- filter_nyc[-train_index, ]

#Getting the average price for all NYC
mu_hat <- mean(train_set$price)
mu_hat

#getting naive rmse
naive_rmse <- RMSE(train_set$price, mu_hat)
naive_rmse

#creating rmse results matrix
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

#Testing Neighbourhood Group Model b_g
mu <- mean(train_set$price) 
ngroup_avgs <- train_set %>% 
  group_by(neighbourhood_group) %>% 
  summarize(b_g = mean(price - mu))

#gathering b_g variable for analysis
predicted_price <- mu + train_set %>% 
  left_join(ngroup_avgs, by='neighbourhood_group') %>%
  .$b_g

#checking rmse of first b_g model and adding to results table
model_1_rmse <- RMSE(predicted_price, train_set$price)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Neighbourhood Group Mode",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

neigh_avgs <- train_set %>% 
  group_by(neighbourhood) %>% 
  summarize(b_n = mean(price - mu))

#gathering b_n variable for analysis
predicted_price <- mu + train_set %>% 
  left_join(neigh_avgs, by='neighbourhood') %>%
  .$b_n

#checking rmse of first b_i model and adding to results table
model_2_rmse <- RMSE(predicted_price, train_set$price)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Neighbourhood Model",
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

#Create Room Type Model with b_t and b_n
type_avgs <- train_set %>% 
  left_join(neigh_avgs, by='neighbourhood') %>%
  group_by(room_type) %>% 
  summarize(b_t = mean(price - mu - b_n))

predicted_price <- train_set %>% 
  left_join(neigh_avgs, by='neighbourhood') %>%
  left_join(type_avgs, by='room_type') %>%
  rowwise()  %>%
  mutate(pred = sum(mu, b_n, b_t)) %>%
  .$pred

model_3_rmse <- RMSE(predicted_price, train_set$price)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Neighbourhood + Room Type Model",
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()


#Creating Regularized Model of b_n and b_t
lambdas <- seq(0, 80, 1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$price)
  b_n <- train_set %>%
    group_by(neighbourhood) %>%
    summarize(b_n = sum(price - mu)/(n()+l))
  b_t <- train_set %>% 
    left_join(b_n, by="neighbourhood") %>%
    group_by(room_type) %>%
    summarize(b_t = sum(price - b_n - mu)/(n()+l))
  predicted_ratings <- 
    train_set %>% 
    left_join(b_n, by = "neighbourhood") %>%
    left_join(b_t, by = "room_type") %>%
    rowwise()  %>%
    mutate(pred = sum( mu, b_n, b_t, na.rm=TRUE)) %>%
    .$pred
  return(RMSE(predicted_ratings, train_set$price))
})
qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Neighbourhood + Room Type Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#Testing the Final Regularized Model 
lambdas <- seq(0, 80, 1)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$price)
  b_n <- train_set %>%
    group_by(neighbourhood) %>%
    summarize(b_n = sum(price - mu)/(n()+l))
  b_t <- train_set %>% 
    left_join(b_n, by="neighbourhood") %>%
    group_by(room_type) %>%
    summarize(b_t = sum(price - b_n - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_n, by = "neighbourhood") %>%
    left_join(b_t, by = "room_type") %>%
    rowwise()  %>%
    mutate(pred = sum( mu, b_n, b_t, na.rm=TRUE)) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$price))
})
qplot(lambdas, rmses) 

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Testing Final Regularized Neighbourhood + Room Type Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#Creating predication comparison of final model to analyze difference of prediction and actual price
lambda <- lambdas[which.min(rmses)]
lambda

mu <- mean(train_set$price)
b_n <- train_set %>%
  group_by(neighbourhood) %>%
  summarize(b_n = sum(price - mu)/(n()+lambda))
b_t <- train_set %>% 
  left_join(b_n, by="neighbourhood") %>%
  group_by(room_type) %>%
  summarize(b_t = sum(price - b_n - mu)/(n()+lambda))
predicted_price <- 
  test_set %>% 
  left_join(b_n, by = "neighbourhood") %>%
  left_join(b_t, by = "room_type") %>%
  rowwise()  %>%
  mutate(pred = sum( mu, b_n, b_t, na.rm=TRUE))
RMSE(predicted_price$pred, test_set$price)

pred_price<-predicted_price$pred
actual_price<-predicted_price$price

qplot(pred_price, actual_price)

#Looking at distribution of neighbourhood groups
ggplot(predicted_price, aes(pred, price, color=neighbourhood_group)) +
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point() +
  ylab("price") +
  xlab("pred") +
  xlim(0,500)

#Looking at distribution of room type
ggplot(predicted_price, aes(pred, price, color=room_type)) +
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point() +
  ylab("Actual Price") +
  xlab("Predicted Price")

#adjusting x axis to show how outliers affect the distribution
ggplot(predicted_price, aes(pred, price, color=room_type)) +
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point() +
  ylab("Actual Price") +
  xlab("Predicted Price") +
  xlim(0,500)

#Creating random sample of 20 predictions
sample_results <- predicted_ratings %>% sample_n(20)%>% mutate(diff=pred-price) %>% 
  select(neighbourhood_group, neighbourhood, room_type, price, pred,diff) %>% 
  arrange(diff) 

print(sample_results)





