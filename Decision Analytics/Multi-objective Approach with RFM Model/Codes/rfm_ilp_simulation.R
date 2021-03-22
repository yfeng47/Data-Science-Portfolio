# @authors: 
# @date: 20200813

# Init
set.seed(1234)
DEBUG=FALSE

# Variables
df <- read.csv('model_data.csv')
avg_cost <- 3.5

# Optimized decision variables
target <- c(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

df$target <- target

# Function that simulate for 1 iteration
simulate_func <- function(df){
  total_profit <- 0
  
  for (i in 1:nrow(df)){
    if(DEBUG){cat('Simulate', i, 'segment\n')}
    total_profit <- total_profit + calc_profit(df, i)
  }
  
  return(total_profit)
}

# Function that returns profit of a customer segment
calc_profit <- function(df, n){
  prob <- rnorm(1, df$Probability[n], sd=.05)
  customer_count <- df$CustomerCount[n]
  expected_revenue <- runif(1, df$ExpectedRevenue[n]*.5, df$ExpectedRevenue[n]*1.5)
  target <- df$target[n]
  profit <- customer_count * (expected_revenue * prob - avg_cost) * target
  return(profit)
}

# Simulate with n iterations
n <- 50000
total_profit <- list()

cat('Running simulation for', n , 'times\n')
print('------------------------------')

for (i in 1:n){
  total_profit <- c(total_profit, simulate_func(df))  # append the simulation result to the list
}

# Analyze the result
total_profit <- as.numeric(total_profit)
total_profit.bar <- mean(total_profit)
hist(total_profit)
abline(v=total_profit.bar, col='red')
text(x=total_profit.bar+100000, y=7000, round(total_profit.bar))
abline(v=1200000, col='blue')
text(x=1300000, y=7000, sum(total_profit>1200000)/50000)
