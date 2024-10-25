pacman::p_load("qgg", "corrplot", "ggplot2", "tidyr", "mlbench", 
               "readr", "data.table", "naniar", "car", "caret",
               "pROC", "xgboost", "dplyr", "Matrix", "e1071")

# Download the Adult data
url <- "https://archive.ics.uci.edu/static/public/2/adult.zip"
destfile <- tempfile()
exdir <- tempdir()
download.file(url=url, destfile=destfile)
unzip(destfile, exdir=exdir)
list.files(exdir, full.names = TRUE) 


# Read the train and test data sets
df_train <- fread(file.path(exdir,"adult.data"), data.table=FALSE)
df_test <- fread(file.path(exdir,"adult.test"), skip=1, data.table=FALSE)

colnames(df_train) <- c("age", "workclass", "fnlwgt", "education", "education_num",
                        "marital_status", "occupation", "relationship", "race",
                        "sex", "capital_gain", "capital_loss", "hours_per_week",
                        "native_country", "income")

colnames(df_test) <- c("age", "workclass", "fnlwgt", "education", "education_num",
                       "marital_status", "occupation", "relationship", "race",
                       "sex", "capital_gain", "capital_loss", "hours_per_week",
                       "native_country", "income")

# Look at the structure of the data frames
str(df_train)
str(df_test)

# Clean the test dataset income column by remove the "."
# Use fixed=TRUE to treat it as a literal string, and not a regular expression:
df_test$income <- gsub(".", "", df_test$income, fixed=TRUE)


colSums(is.na(df_train))
colSums(is.na(df_test))


any(duplicated(rbind(df_train, df_test)))

str(df_train)
# To see the age range where people are more likely to fall into a particular income bracket:
ggplot(df_train, aes(x = age, fill = income)) +
  geom_density(alpha = 0.7) +
  labs(
    title = "Age distribution by income group",
    X = "Age", Y = "Density"
  ) +
  scale_fill_manual(values = c("<=50K" = "#4E7AA7", ">50K" = "#E49343")) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank()
    #,
    #panel.grid.minor = element_blank()
  ) +
  labs(title = "Age Distribution by Income Group",
       x = "Age", y = "Density")


# Change the character variables to factors
df  <- rbind(df_train,df_test)
character_vars <- lapply(df, class) == "character"
df[, character_vars] <- lapply(df[, character_vars], as.factor)
str(df)


# Look at the data to determine whether there may be redundant variables. I check for covariance between variables (correlated features), variables with very low variance (below 1e-5) and categorical variables that consist of only a single value.


## Look at covariance between variables
# Select numerical features
num_vars <- sapply(df_train, is.numeric)

# Compute correlation matrix for numerical variables
# None of the variables appear to be highly correlated
cor_matrix <- cor(df_train[, num_vars], use = "complete.obs")
cor_matrix

# Visualize correlations with a heatmap (if desired)
corrplot(cor_matrix, method = "color", tl.cex = 0.7)

## Look at the variance of numerical variables
# Calculate variance for each numeric column
variances <- apply(df_train[, num_vars], 2, var)

# Find features with low variance (e.g., variance close to 0)
low_variance_features <- names(variances[variances < 1e-5])

## Look for categorical variables with a class that consist of a single value
sapply(df_train[,character_vars], table)

# In native_country only one person is from Holland-Netherlands (spelled "Holand").
# I decide to remove the Holand-Netherlands category.
df$native_country <- as.character(df$native_country)
df <- df[df$native_country != "Holand-Netherlands", ]
table(df$native_country)
df$native_country <- as.factor(df$native_country)

# Remove this observation from the training dataset as well. Since I will use the dimensions of the training set to prepare the training data.
df_train <- df_train[df_train$native_country != "Holand-Netherlands", ]
# Looking at the dimensions of the train dataset, I can see that 1 row has been removed.
dim(df_train)

## Check for variables consisting of a single 
single_value_check <- sapply(df, function(x) length(unique(x)) == 1)


## Define train and test data indices
train <- 1:nrow(df_train) 
# Add the number of rows in the test data to each row number of the training data in order to generate and index for the rows of the test data frame.
test <- (1:nrow(df_test))+ nrow(df_train) 


# Analysis

# Logistic regression

## Define train and test data
df_train <- df[train,]
df_test <- df[test,]

# I look at the structure of the training and test sets to ensure that the structure of the data frame "df" is kept when subsetting.
str(df_train)
str(df_train)

### Models
## Model 1 (formula 1 = f1): include all variables
# R by default assigns the higher factor level as the positive class
glmf1 <- income ~ .
fit_glmf1 <- glm(glmf1, data = df, family = "binomial", subset = train)
# The summary shows that some dummy variables have missing estimates.
summary(fit_glmf1)

# Based on the missing estimates, there seem to be redundant variables in the model which introduce collinearity.
# The education_num and education variables seem to be highly correlated.
alias(fit_glmf1)

# Do a Pearson's Chi-squared test to determine the independence between the "workclass" and "occupation" variables.
# The test shows that there is a strong association (significant p-value and large Chi-square statistic) between these two variables.
chisq.test(table(df_train$workclass, df_train$occupation))

# Remove the education_num and workclass variables and see if it helps.

## Model 2
glmf2 <- income ~ . -education_num -workclass
fit_glmf2 <- glm(glmf2, data = df, family = "binomial", subset = train)
summary(fit_glmf2)

# Looking at the variance inflation factor, it looks like there is no more problematic multicollinearity, since none of the adjusted VIF factors are above 5. Although there could be moderate multicollinearity between the variables relationship, marital_status and sex.
vif(fit_glmf2)

# Take a look at independence between marital_status and relationship variables.
# It could be worth dropping one of these variables, as the Chi-square test shows significant association between them.
chisq.test(table(df_train$marital_status, df_train$relationship))

```

I investigate the data for dependencies. Based on the results I remove the variables *education_num*, *workclass* and *relationship* from the model.
```{r}
#| results: hide

## Model 3
glmf3 <- income ~ . -education_num -workclass -relationship
fit_glmf3 <- glm(glmf3, data = df, family = "binomial", subset = train)
fit_glmf3 <- glm(glmf3, data = df_train, family = "binomial")
summary(fit_glmf3)
vif(fit_glmf3)

# Predict the probabilities using the logistic regression model
probs_glmf3 <- predict(fit_glmf3, newdata = df_test, type = "response")

# Predict the class based on the calculated probabilities
# Set predicted class to ">50K" for observations with probabilities greater than 0.5
pred_glmf3 <- rep("<=50K", 16281)
pred_glmf3[probs_glmf3 > 0.5] <- ">50K" 
table(pred_glmf3, df_test$income)

contrasts(df$income)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glmf3 <- confusionMatrix(table(pred_glmf3, df_test$income), positive = ">50K")

specificity_glmf3 <- confmatrix_glmf3$byClass["Specificity"]
accuracy_glmf3 <- as.numeric(confmatrix_glmf3$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_glmf3 <- round(as.numeric(confmatrix_glmf3$byClass["Pos Pred Value"]), 3)

# Draw the ROC curve and extract area under the curve
y <- as.numeric(df_test$income)-1
roc_glmf3 <- roc(y, probs_glmf3)
auc_glmf3 <- roc_glmf3$auc
plot(roc_glmf3)

# Notes to myself with regards to the ROC curve:
# On the leftmost point of the ROC curve, the threshold is set to a very high value (1), thus only the highest possible predicted probabilities are classified as positive (>50K in this case). This results in almost all predictions being classified as <=50K.
# This leads to 0 sensitivity, as the model does not identify positive cases.
# The pROC package plots Specificity on the x-axis. Therefore the x-axis goes from 1 on the left axis to 0 on the right of the x-axis.
```

After removing variables that likely are collinear, the model still returns a warning that "glm.fit: fitted probabilities numerically 0 or 1 occurred". In order to determine what may cause the perfect separation, I look at the summary of the fitted model to see if there may be extremely large coefficients or standard errors, which can indicate separation. *education* category "Preschool" and *native_country* category "Outlying-US(Guam_USVI-etc)" have the the biggest estimate and standard error. A look at the contingency tables between income and these variables, shows only one observation for each of these categories. I decide to move the "Preschool" observation to 1st-4th grade and the Outlying-US(Guam_USVI-etc) observation to the "?" category for native_country.

```{r}
#| results: hide

# Save the summary of the model fit as an object
summary_glmf3 <- summary(fit_glmf3)

# Extract coefficients (includes estimates, standard errors, etc.)
coefficients <- summary_glmf3$coefficients

# View the coefficients
# coefficients is a matrix with columns: Estimate, Std. Error, z value, Pr(>|z|)
print(coefficients)

# # To extract the row with the maximum standard error:
# max_std_error_row <- coefficients[which.max(coefficients[, "Std. Error"]), ]
# print(max_std_error_row)

# To order the coefficients by standard error:
ordered_by_std_error <- coefficients[order(coefficients[, "Std. Error"], decreasing = TRUE), ]


# To order the coefficients by estimate:
ordered_by_estimate <- coefficients[order(abs(coefficients[, "Estimate"]), decreasing = TRUE), ]

## Look at contingency tables between income and native_country and education.
table(df$income, df$native_country)
table(df$income, df$education)

# Combine the "Preschool" observation with 1st-4th grade and the Outlying-US(Guam_USVI-etc) observation with the "?" category for native_country
# Drop the unused factor levels
# Look at the new levels of education and native_country
df$education[df$education == "Preschool"] <- "1st-4th"
df$education <- droplevels(df$education)
table(df$education)

# df$native_country[df$native_country == "Outlying-US(Guam-USVI-etc)"] <- "?"
# df$native_country <- droplevels(df$native_country)
# table(df$native_country)
```

Run a logistic regression model without the variables and after removing the categories with a single observation in the contingency table between income and the predictors.

```{r}
#| results: hide

# Create train and test sets
df_train <- df[train, ]

# Create a contingency table of income and native_country
contingency_table_native_country <- table(df_train$income, df_train$native_country)

# Find the native_country categories with fewer than 10 observations in any income row
low_count_country <- apply(contingency_table_native_country, 2, function(x) any(x < 10))

# Extract the countries that meet the condition
low_count_countries <- names(low_count_country[low_count_country])

# Sum the observations for each native_country across income categories
# country_totals <- colSums(contingency_table_native_country)


# Move all low count countries to a new category "other"
df$native_country <- as.character(df$native_country)
df$native_country[df$native_country %in% low_count_countries] <- "Other"
df$native_country <- as.factor(df$native_country)

# Move occupations with low counts to category "?"
# Armed-Forces and Priv-house-serv gets moved to "?"
df$occupation <- as.character(df$occupation)
df$occupation[df$occupation %in% c("Armed-Forces", "Priv-house-serv")] <- "?"
df$occupation <- as.factor(df$occupation)

# Married - Armed Forces spouse -> move to Married-civ-spouse
table(df$income, df$marital_status)
df$marital_status <- as.character(df$marital_status)
df$marital_status[df$marital_status %in% "Married-AF-spouse"] <- "Married-civ-spouse"
df$marital_status <- as.factor(df$marital_status)


# Create train and test sets
df_train <- df[train, ]
df_test <- df[test,]

## Model 4
glmf4 <- income ~ . -education_num -workclass -relationship
fit_glmf4 <- glm(glmf4, data = df_train, family = "binomial")
summary_glmf4 <- summary(fit_glmf4)
vif(fit_glmf4)

summary_glmf4$coefficients[order(summary_glm4$coefficients[, "Std. Error"], decreasing = TRUE), ]


# Predict the probabilities using the logistic regression model
probs_glmf4 <- predict(fit_glmf4, newdata = df_test, type = "response")

# Predict the class based on the calculated probabilities
# Set predicted class to ">50K" for observations with probabilities greater than 0.5
pred_glmf4 <- rep("<=50K", 16281)
pred_glmf4[probs_glmf4 > 0.5] <- ">50K" 
table(pred_glmf4, df_test$income)

contrasts(df$income)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glmf4 <- confusionMatrix(table(pred_glmf4, df_test$income), positive = ">50K")

specificity_glmf4 <- confmatrix_glmf4$byClass["Specificity"]
accuracy_glmf4 <- as.numeric(confmatrix_glmf4$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_glmf4 <- round(as.numeric(confmatrix_glmf4$byClass["Pos Pred Value"]), 3)


# I still get the same warning: "glm.fit: fitted probabilities numerically 0 or 1 occurred" 
```

Somehow the issue with perfect separation is still not resolved. So, I discuss this problem with Peter Sørensen and he suggests that I should try using a design/model matrix.

```{r}


# Convert income variable to binary outcome
# df$income <- ifelse(df$income == "<=50K", 0, 1)
# df$income <- as.factor(df$income)
# table(df$income)

# Define train/test data
df_train <- df[train,]
df_test <- df[test,]

# Create a model matrix using the model.matrix() function, which converts factors into dummy variables and represent all predictor variables in a matrix format.
# Remove the intercept, because model.matrix() by default includes an intercept column.

X_train <- model.matrix(income ~ . -1 -education_num -workclass -relationship, data = df_train)
fit_glmf <- glm(df_train$income ~ X_train, family = "binomial")

summary_glmf <- summary(fit_glmf)
summary_glmf$coefficients[order(summary_glmf$coefficients[, "Std. Error"], decreasing = TRUE), ]
summary_glmf$coefficients[order(summary_glmf$coefficients[, "Estimate"], decreasing = TRUE), ]

# From the summary, some of the occupation categories have the highest std. error and marital status have the largest estimate. Look at contingency tables for these variables
table(df$income, df$occupation)
# Priv-house-serv and Armed-Forces have only few observations, move these to "?"
df$occupation <- as.character(df$occupation)
df$occupation[df$occupation %in% c("Priv-house-serv", "Armed-Forces")] <- "?"
df$occupation <- as.factor(df$occupation)

table(df$income, df$marital_status)
# Married - Armed Forces spouse -> move to Married-civ-spouse
df$marital_status <- as.character(df$marital_status)
df$marital_status[df$marital_status %in% "Married-AF-spouse"] <- "Married-civ-spouse"
df$marital_status <- as.factor(df$marital_status)

table(df$income, df$native_country)
```

Try using regularization to solve the problem of overfitting and separation issues.

```{r}
# Load necessary libraries
library(glmnet)

# I have read the data in from the beginning. Thus without removing categories with low levels


# Prepare data
# Assuming df is your dataframe and you are using 'income' as the response variable
# Convert 'income' to a binary numeric outcome (0/1)
df$income_binary <- ifelse(df$income == "<=50K", 0, 1)

df_train <- df[train,]
df_test <- df[test,]

# Extract predictor variables as a matrix and response variable as a vector
# Exclude the 'income' and other non-numeric columns if needed
X_train <- model.matrix(income_binary ~ . - income, data = df_train)[, -1]  # Create matrix of predictors, excluding the intercept
y_train <- df_train$income_binary  # Binary outcome

X_test <- model.matrix(income_binary ~ . - income, data = df_test)[, -1] 
y_test <- df_test$income_binary

# Fit Ridge regression (alpha = 0 means Ridge in glmnet)
ridge_model <- glmnet(X_train, y_train, family = "binomial", alpha = 0)

# Print a summary of the model
print(ridge_model)

# Use cross-validation to find the best lambda (regularization parameter)
cv_ridge <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0)

# Plot the cross-validation results
plot(cv_ridge)

# Find the best lambda that minimizes the cross-validated error
best_lambda <- cv_ridge$lambda.min
print(best_lambda)

# Refit the model using the best lambda
ridge_model_best <- glmnet(X_train, y_train, family = "binomial", alpha = 0, lambda = best_lambda)

# Coefficients of the final model
coef(ridge_model_best)

# Make predictions on the same dataset (or on a test set if available)
# This returns the predicted probabilities
predicted_probabilities <- predict(ridge_model_best, X_test, type = "response")

# Convert probabilities to class predictions (optional)
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Confusion matrix to evaluate performance
table(Predicted = predicted_classes, Actual = y)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glm_ridge <- confusionMatrix(table(predicted_classes, y_test), positive = "1")

specificity_glm_ridge <- confmatrix_glm_ridge$byClass["Specificity"]
accuracy_glm_ridge <- as.numeric(confmatrix_glm_ridge$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_glm_ridge <- round(as.numeric(confmatrix_glm_ridge$byClass["Pos Pred Value"]), 3)


# Load necessary libraries
library(ggplot2)

# Extract the coefficients for the best lambda
best_lambda <- cv_ridge$lambda.min
coefficients <- coef(ridge_model_best, s = best_lambda)

# Convert the coefficient matrix to a data frame for easier plotting
coefficients_df <- as.data.frame(as.matrix(coefficients))
coefficients_df$Variable <- rownames(coefficients_df)
colnames(coefficients_df) <- c("Coefficient", "Variable")

# Remove the intercept from the plot (optional)
coefficients_df <- coefficients_df[coefficients_df$Variable != "(Intercept)", ]

# Plot the coefficients using ggplot2
ggplot(coefficients_df, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip the coordinates to make the plot horizontal
  labs(title = "Variable Importance in Ridge Regression",
       x = "Variables", y = "Coefficient") +
  theme_minimal()


```



### XGBoost

Since XGBoost is a tree-based algorithm, and therefore should not be particularly sensitive to collinear features, I include all variables in the first model. I train a subsequent XGBoost model where I remove three variables found to be redundant ([see Logistic Regression](#logistic-regression)).
  
  XGBoost requires numerical inputs so I need to preprocess the data.
  
  ```{r}
  #| results: hide
  
  # Convert income variable to binary outcome
  df$income <- ifelse(df$income == "<=50K", 0, 1)
  table(df$income)
  
  # Define train/test data
  train_df <- df[train,]
  test_df <- df[test,]
  ```
  
  The first XGBoost model includes all the variables.
  
  ```{r}
  #| results: hide
  
  ## XGBoost model 1: includes all variables
  ## Create model matrix for xgboost and define the reponse variable
  # One-hot encode categorical variables - i.e. each category for each categorical variable becomes a binary variable coded 0/1
  # The model.matrix() function automatically detects and encodes all categorical variables.
  # The intercept is remove from the model matrix: "-1"
  train_matrix <- model.matrix(income ~ . -1, data = train_df)
  train_label <- train_df$income
  table(train_label)
  
  test_matrix <- model.matrix(income ~ . -1, data = test_df)
  test_label <- test_df$income
  
  # Convert the data to DMatrix, which is XGBoost's optimized data structure
  train_xgb <- xgb.DMatrix(data = train_matrix, label = train_label)
  test_xgb <- xgb.DMatrix(data = test_matrix, label = test_label)
  
  ## Train the XGBoost model
  # Set the XGBoost parameters
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.1,  # learning rate
    max_depth = 6,
    gamma = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "error"
  )
  
  # Train the model
  # Prevent the output from being printed in the rendered html document by using invisible()
  invisible({
    fit_xgb1 <- xgb.train(
      params = params, 
      data = train_xgb, 
      nrounds = 100, 
      watchlist = list(train = train_xgb, eval = test_xgb), 
      early_stopping_rounds = 10,
      print_every_n = 10
    )
  })
  
  # Feature importance
  # Could be interesting to compare with the summary of logistic regression
  importance <- xgb.importance(feature_names = colnames(train_xgb), model = fit_xgb1)
  xgb.plot.importance(importance)
  
  ## Make predictions
  probs_xgb1 <- predict(fit_xgb1, newdata = test_xgb)
  pred_xgb1 <- ifelse(probs_xgb1 > 0.5, 1, 0)
  
  ## Confusion matrix
  confmatrix_xgb1 <- confusionMatrix(factor(pred_xgb1), factor(test_label), positive = "1")
  sensitivity_xgb1 <- confmatrix_xgb1$byClass["Sensitivity"] # Recap: sensitivity is the recall for the positive class
  specificity_xgb1 <- confmatrix_xgb1$byClass["Specificity"] # Recap: specificity is the recall for the negative class
  
  # Extract precision from the confusion matrix.
  # Precision = TP/(TP + FP)
  # Precision is given as "Pos Pred Value" in the caret package
  precision_xgb1 <- round(as.numeric(confmatrix_xgb1$byClass["Pos Pred Value"]), 3)
  
  # To get 95% confidence interval on precision:
  # Extract the table from the confusionMatrix() results
  cm_matrix_xgb1 <- confmatrix_xgb1$table
  
  # Extract the number of True Positives (TP) and False Positives (FP)
  TP <- cm_matrix_xgb1[2, 2]  # True Positives (predicted positive and actual positive)
  FP <- cm_matrix_xgb1[2, 1]  # False Positives (predicted positive and actual negative)
  
  # Calculate precision (TP / (TP + FP))
  precision_value <- TP / (TP + FP)
  
  # Perform a binomial test to get the confidence interval
  precision_ci_xgb1 <- binom.test(TP, TP + FP)$conf.int
  
  # Print precision and its confidence interval
  print(paste("Precision:", round(precision_value, 3)))
  print(paste("95% Confidence Interval for Precision:", round(precision_ci_xgb1[1], 3), "-", round(precision_ci_xgb1[2], 3)))
  
  # Draw the ROC curve and extract area under the curve
  y <- test_label
  roc_xgb1 <- roc(y, probs_xgb1)
  auc_xgb1 <- roc_xgb1$auc
  plot(roc_xgb1)
  ```
  
  In the second XGBoost model I exclude the collinear variables, because I am curious to see how it may influence the accuracy and AUC of the XGBoost model.
  
  ```{r}
  #| results: hide
  
  ## XGBoost model 2: exclude collinear variables
  ## Create model matrix for xgboost and define the reponse variable
  # One-hot encode categorical variables - i.e. each category for each categorical variable becomes a binary variable coded 0/1
  # The model.matrix() function automatically detects and encodes all categorical variables.
  # The intercept is removed from the model matrix: "-1"
  train_matrix2 <- model.matrix(income ~ . -1  -education_num -workclass -relationship, data = train_df)
  train_label <- train_df$income
  
  test_matrix2 <- model.matrix(income ~ . -1  -education_num -workclass -relationship, data = test_df)
  test_label <- test_df$income
  
  # Convert the data to DMatrix, which is XGBoost's optimized data structure
  train_xgb2 <- xgb.DMatrix(data = train_matrix2, label = train_label)
  test_xgb2 <- xgb.DMatrix(data = test_matrix2, label = test_label)
  
  ## Train the XGBoost model
  # Set the XGBoost parameters
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.1,  # learning rate
    max_depth = 6,
    gamma = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "error"
  )
  
  # Train the model
  invisible({
    fit_xgb2 <- xgb.train(
      params = params, 
      data = train_xgb2, 
      nrounds = 100, 
      watchlist = list(train = train_xgb2, eval = test_xgb2), 
      early_stopping_rounds = 10, 
      print_every_n = 10
    )
  })
  
  ## Make predictions
  probs_xgb2 <- predict(fit_xgb2, newdata = test_xgb2)
  pred_xgb2 <- ifelse(probs_xgb2 > 0.5, 1, 0)
  
  # Confusion matrix and accuracy
  confmatrix_xgb2 <- confusionMatrix(factor(pred_xgb2), factor(test_label), positive = "1")
  
  confmatrix_xgb2$byClass["Specificity"]
  
  # Draw the ROC curve and extract area under the curve
  y <- test_label
  roc_xgb2 <- roc(y, probs_xgb2)
  roc_xgb2$auc
  # plot(roc_xgb2)
  ```
  
  In the third model I exclude collinear variables and remove duplicates in the data (although I think it may be relevant to keep duplicates).
  
  ```{r}
  #| results: hide
  
  ## XGBoost model 3: exclude collinear variables and remove duplicates in the data
  
  isdup <- duplicated(df)
  df_wo_dup <- df[!isdup,]
  
  ## Define train and test data
  # Ensure train_df and test_df contains only rows originally in df_train and df_test
  train_df_wo_dup <- df_wo_dup[rownames(df_wo_dup) %in% rownames(df_train), ]
  test_df_wo_dup <- df_wo_dup[rownames(df_wo_dup) %in% rownames(df_test),]
  
  ## Create model matrix for xgboost and define the reponse variable
  # One-hot encode categorical variables - i.e. each category for each categorical variable becomes a binary variable coded 0/1
  # The model.matrix() function automatically detects and encodes all categorical variables.
  # The intercept is removed from the model matrix: "-1"
  train_matrix3 <- model.matrix(income ~ . -1  -education_num -workclass -relationship, data = train_df_wo_dup)
  train_label_wo_dup <- train_df_wo_dup$income
  
  test_matrix3 <- model.matrix(income ~ . -1  -education_num -workclass -relationship, data = test_df_wo_dup)
  test_label_wo_dup <- test_df_wo_dup$income
  
  # Convert the data to DMatrix, which is XGBoost's optimized data structure
  train_xgb3 <- xgb.DMatrix(data = train_matrix3, label = train_label_wo_dup)
  test_xgb3 <- xgb.DMatrix(data = test_matrix3, label = test_label_wo_dup)
  
  ## Train the XGBoost model
  # Set the XGBoost parameters
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.1,  # learning rate
    max_depth = 6,
    gamma = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "error"
  )
  
  # Train the model
  fit_xgb3 <- xgb.train(
    params = params, 
    data = train_xgb3, 
    nrounds = 100, 
    watchlist = list(train = train_xgb3, eval = test_xgb3), 
    early_stopping_rounds = 10, 
    print_every_n = 10
  )
  
  ## Make predictions
  probs_xgb3 <- predict(fit_xgb3, newdata = test_xgb3)
  pred_xgb3 <- ifelse(probs_xgb3 > 0.5, 1, 0)
  
  # Confusion matrix and accuracy
  confmatrix_xgb3 <- confusionMatrix(factor(pred_xgb3), factor(test_label_wo_dup), positive = "1")
  
  # Draw the ROC curve and extract area under the curve
  y <- test_label_wo_dup
  roc_xgb3 <- roc(y, probs_xgb3)
  roc_xgb3$auc
  auc_xgb3 <- as.numeric(roc_xgb3$auc)
  # plot(roc_xgb3)
  ```
  
  ### Support Vector Classifier
  
  The support vector classifier (SVC) is a method that constructs a set of hyperplanes that separates the training data into two classes. Collinearity can degrade the performance of the classifier when using a linear kernel. To look at the influence of redundant variables in the model, I will run an SVC model with and without the redundant variables using a linear kernel and a support vector machine model with a radial kernel where all variables are included in the model.
  
  I will also perform regularization by playing around with the C tuning parameter, which allows changing the number and severity of the violations to the margin. A smaller C value increases the margin size by allowing more violations, while a larger C value reduces the margin and penalizes misclassifications more severely.
  
  <!-- How the C (cost) tuning parameter works: -->
    
    <!-- C Tuning Parameter: The C parameter in SVC controls the degree of regularization. -->
    
    <!-- A small C (more regularization) allows a larger margin with more violations (misclassified points), making the classifier more tolerant to errors, which can help generalize better. -->
    
    <!-- A large C (less regularization) reduces the margin and penalizes violations more severely, making the classifier focus on correctly classifying all points, which may lead to overfitting. -->
    
    I keep the response variable as numeric and convert it to a binary factor for classification.
  
  ```{r}
  #| results: hide
  
  # Change the income variable to a factor
  df$income <- as.factor(df$income)
  
  # Define train/test data
  train_df <- df[train,]
  test_df <- df[test,]
  ```
  
  The first SVC model includes all the predictor variables.
  
  ```{r}
  #| results: hide
  
  ## Model 1
  # Train a Support Vector Classifier model with probability estimation
  # runtime_svc1 <- system.time({
  # fit_svc1 <- svm(income ~ ., data = train_df, kernel = "linear", cost = 1, probability = TRUE)
  # })
  
  
  runtime_svc1 <- system.time({
    fit_svc1 <- svm(income ~ ., data = train_df, kernel = "linear", cost = 1)
  })
  runtime_svc1["elapsed"]
  
  
  ## Predictions on the test set
  # Estimate the predicted class as well as probabilities for a certain class
  pred_svc1 <- predict(fit_svc1, test_df, probability = TRUE)
  
  # Evaluate the model's performance
  confmatrix_svc1 <- confusionMatrix(pred_svc1, test_df$income, positive = "1")
  
  # Extract accuracy
  accuracy_svc1 <- as.numeric(confmatrix_svc1$overall["Accuracy"])
  
  # Extract precision from the confusion matrix.
  # Precision = TP/(TP + FP)
  # Precision is given as "Pos Pred Value" in the caret package
  precision_svc1 <- round(as.numeric(confmatrix_svc1$byClass["Pos Pred Value"]), 3)
  
  ## Draw the ROC curve and extract AUC
  # Probalities can be extracted from pred_svc1 object where it is stored as an attribute
  y <- test_df$income
  probs_attr_svc1 <- attr(pred_svc1, "probabilities")
  # Extract probabilities for the positive class label
  probs_svc1 <- probs_attr_svc1[, "1"] 
  roc_svc1 <- roc(y, probs_svc1)
  auc_svc1 <- as.numeric(roc_svc1$auc)
  ```
  
  In the second SVC model I remove the variables *education_num*, *workclass* and *relationship* that were shown to be redundant ([see Logistic Regression](#logistic-regression)).
    
    ```{r}
    #| results: hide
    
    ## SVC model 2
    # Train the SVC model and include the calculation of probabilities
    svcf2 <- income ~ . - education_num - workclass - relationship
    
    # Train the SVC model at different values of the C tuning parameter
    # I use the built-in tune() function in the e1071 library
    set.seed(123)
    runtime_tune_svc2 <- system.time({
      tune_svc2 <- tune(svm, svcf2, data = train_df, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), probability = TRUE)
    })
    
    ### OBS do this VERY IMPORTANT #######
    runtime_tune_svc2 <- runtime_tune_svc1
    
    print(paste("Probability prediction with tuning runtime:", round(runtime_tune_svc2["elapsed"]/60, 2), "minutes"))
    # [1] "Probability prediction with tuning runtime: 31.04 minutes"
    
    # This computation takes quite some time, so I am saving the results.
    saveRDS(tune_svc2, file = "tune_svc2.rds")
    
    # Access the cross-validation errors for each of the models
    summary(tune_svc2)
    
    # Extract the best model
    bestmod_svc2 <- tune_svc2$best.model
    
    ## Predict the class and probabilities using the best model
    pred_svc2 <- predict(bestmod_svc2, test_df, probability = TRUE)
    
    ## Evaluate the model's performance
    confmatrix_svc2 <- confusionMatrix(pred_svc2, test_df$income, positive = "1")
    
    # Extract accuracy
    accuracy_svc2 <- as.numeric(confmatrix_svc2$overall["Accuracy"])
    
    # Extract precision from the confusion matrix.
    # Precision = TP/(TP + FP)
    # Precision is given as "Pos Pred Value" in the caret package
    precision_svc2 <- round(as.numeric(confmatrix_svc2$byClass["Pos Pred Value"]), 3)
    
    # Draw the ROC curve and extract AUC
    # Probalities can be extracted from the pred_svc2 object where it is stored as an attribute
    y <- test_df$income
    probs_attr_svc2 <- attr(pred_svc2, "probabilities")
    # Extract probabilities for the positive class label
    probs_svc2 <- probs_attr_svc2[, "1"] 
    roc_svc2 <- roc(y, probs_svc2)
    auc_svc2 <- as.numeric(roc_svc2$auc)
    ```
    
    The third support vector mo
    
    ## Results
    
    The results of the different analyses are printed as ROC curves.
    
    ```{r}
    #| fig-cap: ROC curves for each of the different analyses methods.
    
    # Plot the first ROC curve: the final logistic regression model
    # Setting legacy.axes = TRUE changes the X-axis to the proportion of true negatives (1-Specificity) as opposed to the false positive rate (FPR)
    plot(roc_glmf3, col = "blue", lwd = 2, legacy.axes = TRUE)
    # Add the ROC curve for XGBoost
    lines(roc_xgb2, col =  "red", lwd = 2, legacy.axes = TRUE)
    # Add a legend
    legend("bottomright",
           legend = c(paste("Logistic Regression AUC:", round(auc_glmf3, 3)),
                      paste("XGBoost AUC:", round(auc_xgb1, 3))),
           col = c("blue", "red"),
           lwd = 2)
    ```
    