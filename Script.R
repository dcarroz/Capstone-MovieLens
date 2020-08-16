###########################################################################
## Title: "MovieLens Project"
## Author: "Diógenes Carroz"
## Date: "10-08-2020"
###########################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

###########################################################################
## Create edx set, validation set
###########################################################################

# Note: this process could take a couple of minutes
if(!(exists("edx") & exists("validation"))) {
    message("Create edx and validation sets")
  
    # MovieLens 10M dataset:
    # https://grouplens.org/datasets/movielens/10m/
    # http://files.grouplens.org/datasets/movielens/ml-10m.zip

    dl <- tempfile()
    download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

    ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

    movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
    colnames(movies) <- c("movieId", "title", "genres")
    movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

    movielens <- left_join(ratings, movies, by = "movieId")

    # Validation set will be 10% of MovieLens data
    set.seed(1, sample.kind="Rounding")
    # if using R 3.5 or earlier, use `set.seed(1)` instead
    test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
    edx <- movielens[-test_index,]
    temp <- movielens[test_index,]

    # Make sure userId and movieId in validation set are also in edx set
    validation <- temp %>% 
        semi_join(edx, by = "movieId") %>%
        semi_join(edx, by = "userId")

    # Add rows removed from validation set back into edx set
    removed <- anti_join(temp, validation)
    edx <- rbind(edx, removed)

    rm(dl, ratings, test_index, temp, movies, movielens, removed)
}

###########################################################################
## Here my code begins                 
###########################################################################

options(digits = 6)

###########################################################################
## RMSE function 
##
## Calculate RMSE (residual mean squared error)
###########################################################################
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
}

###########################################################################
## create_sets function
##
## Create Train set and test set
###########################################################################
create_sets <- function(data) {
  
    # date represents the duration in weeks of the rating record in the data 
    data <- mutate(data, date = round_date(as_datetime(timestamp), unit = "week"))
    data <- data %>% select(rating, userId, movieId, date, genres)
  
    # test set will be 10% of data
    set.seed(1, sample.kind="Rounding")
    # if using R 3.5 or earlier, use `set.seed(1)` instead
    test_index <- createDataPartition(y = data$rating, times = 1, p = 0.1, list = FALSE)
    train_set <- data[-test_index,]
    temp <- data[test_index,]
  
    # Make sure userId, movieId, date and genres in test set are also in train set
    test_set <- temp %>% 
        semi_join(train_set, by = "movieId") %>%
        semi_join(train_set, by = "userId") %>%
        semi_join(train_set, by = "date") %>%
        semi_join(train_set, by = "genres")
  
    # Add rows removed from test set back into train set
    removed <- anti_join(temp, test_set)
    train_set <- rbind(train_set, removed)
    
    return(list(train = train_set,test = test_set))
}  

###########################################################################
## predicter_ratings function
##
## Make Predicted Ratings for Movielens
###########################################################################
predicter_ratings <- function(train_data, test_data, lambda) {
    ## Y_(u,i) = mu + b_i + b_u + f(d_u,i) + g_(u,i)
  
    # ratings' average
    mu <- mean(train_data$rating)
  
    # Regularized Movie Effect
    b_i <- train_data %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n() + lambda))
  
    # Regularized User Effect
    b_u <- train_data %>% 
        left_join(b_i, by="movieId") %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))
        
    # Regularized date (timestamp) effect
    b_d <- train_data %>%
        left_join(b_i, by="movieId") %>%
        left_join(b_u, by="userId") %>%
        group_by(date) %>%
        summarize(b_d = sum(rating - b_u - b_i - mu)/(n() + lambda))

    # Regularized genres effect
    b_g <- train_data %>%
        left_join(b_i, by="movieId") %>%
        left_join(b_u, by="userId") %>%
        left_join(b_d, by="date") %>%
        group_by(genres) %>%
        summarize(b_g = sum(rating - b_d - b_u - b_i - mu)/(n() + lambda))  
  
    # Predicted Ratings with Regularized Movie + User + date + genres Effect Model
    predicted_ratings <- test_data %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_d, by = "date") %>%
        left_join(b_g, by = "genres") %>%
        mutate(pred = mu + b_i + b_u + b_d + b_g) %>%
        .$pred
  
    return(predicted_ratings)
}

###########################################################################
## training_lambda function
##
## Calculate the best lambda for prediction
###########################################################################
training_lambda <- function(train_data, test_data, lbmin=1, lbmax=10, s=1, ite=1) {
    # Perform "ite" iterations to calculate the best lambda between 
    # the values "lbmin" and "lbmax"
  
    s <- max(s,1) 
    ite <- max(ite,1)
    br <- 99999
    bl <- 0
    lambdas <- seq(lbmin, lbmax, s)
    for(i in 1:ite) {
    
        message("   Iteration ", i)
    
        rmses <- sapply(lambdas, function(l) {
            r <- RMSE(predicter_ratings(train_data, test_data, l) ,test_data$rating)
            message("      RMSE = ", round(r,6), "  Lambda = ",l)
            r
        })
    
        minr <- min(rmses)
        lminr <- lambdas[which.min(rmses)]
    
        # Determine the best lambda
        if(minr < br) {
            br <- minr
            bl <- lminr
        }
        
        message("   ----------------------------------------")
        message("   Minimum RMSE = ", round(br,6), "  Lambda = ",bl)
        message("   ----------------------------------------")
    
        # Cut the gap in half and fix the range around the best lambda
        s <- s/2    
        lambdas <- c(bl - s, bl + s)
    }
    return(bl)
}

###########################################################################
## pred_model function
##
## Build the rating prediction.
###########################################################################
pred_model <- function() {
  
    message("1. Split edx in train and test set")
  
    # Separate edx into train and test set 
    sets <- create_sets(edx)
    train_set <- sets$train
    test_set <- sets$test
    rm(sets)

    message("2. Training model to get the best lambda")
  
    # Calculate the best lambda
    lambda <- training_lambda(train_set, test_set, 4, 6, 1, 3)
  
    message("3. Calculate predictions for Validation data")
  
    # Make sure date and genres in validation set are also in train set 
    temp <- mutate(validation, date = round_date(as_datetime(timestamp), unit = "week"))
    validation <- temp %>% 
        semi_join(train_set, by = "date") %>%
        semi_join(train_set, by = "genres")
    removed <- anti_join(temp, validation)
    train_set <- rbind(train_set, removed)
    rm(temp, removed)
  
    # Return Calculate predictions
    predicter_ratings(train_set, validation, lambda)  
}

###########################################################################
## Calculate predictions
###########################################################################
message("Calculate predictions")

predicted_ratings <- pred_model()

###########################################################################
## Calculate RMSE
###########################################################################
message("Calculate RMSE")

rmse <- RMSE(predicted_ratings, validation$rating)
print(rmse)

## End Code
