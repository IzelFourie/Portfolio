# glmnet: Ridge regression



pacman::p_load("qgg", "corrplot", "ggplot2", "tidyr", "mlbench", 
               "readr", "data.table", "naniar", "car", "caret",
               "xgboost", "dplyr", "Matrix", "e1071",
               "glmnet", "ROCR", "pROC")

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

# Clean the test dataset income column by removing the "."
# Use fixed=TRUE to treat it as a literal string, and not a regular expression:
df_test$income <- gsub(".", "", df_test$income, fixed=TRUE)


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

# Look for missing values
colSums(is.na(df_train))
colSums(is.na(df_test))

# Look at duplicated values
any(duplicated(rbind(df_train, df_test)))

# Merge the training and test datasets
# Change the character variables to factors
df  <- rbind(df_train,df_test)
character_vars <- lapply(df, class) == "character"
str(df)

# Look at the distribution of the numeric columns
boxplot(df$capital_gain, main = "Boxplot of Capital Gain")

# Identify numeric colummns
numeric_columns <- df[, sapply(df, is.numeric)]
par(mfrow = c(2, 3))  # Set up the plotting area for multiple plots (2 rows, 3 columns)
lapply(names(numeric_columns), function(col) {
  boxplot(numeric_columns[[col]], main = col, ylab = col)
})
par(mfrow = c(1, 1))

# Look at the distribution of capital gain and capital loss
# (In the future could look at whether numerical variables are normally distributed)
summary(df$capital_gain)
summary(df$capital_loss)

# Change capital gain and capital loss to binary variables
df$capital_gain <- ifelse(df$capital_gain > 0, 1, 0)
df$capital_gain <- as.factor(df$capital_gain)

df$capital_loss <- ifelse(df$capital_loss > 0, 1, 0)
df$capital_loss <- as.factor(df$capital_loss)

# Look at the data to determine whether there may be redundant variables.
# I check for covariance between variables (correlated features), variables with very low variance (below 1e-5) and the number of events/observations per outcome level in relation to the predictors.

## Look at covariance between variables
# Select numerical features in the training data
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

## Check for variables consisting of a single value
single_value_check <- sapply(df, function(x) length(unique(x)) == 1)

#### Look at number of observations for the different outcomes for predictor variable classes
# I want to have at least 10 observations for each level of the outcome variable for each of the predictor variable classes/categories

## Look at the number of observations for different classes of categorical variables
sapply(df_train[,character_vars], table)

# Create contingency tables between each character variable and df$income
character_vars_pred <- setdiff(names(df_train)[sapply(df_train, is.character)], "income")
contingency_tables <- lapply(character_vars_pred, function(var) {
  table(df_train[[var]], df_train$income)
})

# Name each table by the character variable
names(contingency_tables) <- character_vars_pred

# View the contingency tables
contingency_tables

# Above shows that the variables workclass, education, occupation and native_country have categories with few observations
# I decide to combine some of the categories
# Add preschool to 12th grade together and call it pre_to_sec_school

## workclass and occupation seem to be two very related variables
# Do a Pearson's Chi-squared test to determine the independence between the "workclass" and "occupation" variables.
# Use the Monte Carlo simulation, to solve the problem of some categories with very small counts.
# The test shows that there is a strong association (significant p-value and large Chi-square statistic) between these two variables.
# Will therefore only include one of them in the analysis model
chisq.test(table(df_train$workclass, df_train$occupation), simulate.p.value = TRUE)

# Of course working Without_pay and Never_worked is very unlikely to result in an income above >50K. I am merging these categories with the category "?"
df$workclass[df$workclass %in% c("Without-pay", "Never-worked")] <- "?" 
table(df$income, df$workclass)

## education
# Add Preschool, to other grades up to 9th grade and call it pre-primary-school
df$education[df$education %in% c("Preschool", "1st-4th", "5th-6th", "7th-8th", "9th")] <- "pre-primary-school"
table(df$income, df$education)

## Occupation
# Move Armed-Forces and Priv-house-serv to "?"
df$occupation[df$occupation %in% c("Armed-Forces", "Priv-house-serv")] <- "?"
table(df$income, df$occupation)

## native_country
# Create a contingency table of income and native_country
contingency_table_native_country <- table(df_train$income, df_train$native_country)

# Find the native_country categories with fewer than 10 observations in any income row
low_count_country <- apply(contingency_table_native_country, 2, function(x) any(x < 10))

# Extract the countries that meet the condition
low_count_countries <- names(low_count_country[low_count_country])

# Move all low count countries to a new category "other"
df$native_country[df$native_country %in% low_count_countries] <- "Other"

# Create a binary variable for income
df$income <- ifelse(df$income == ">50K", 1, 0)
df$income <- as.factor(df$income)

# Change all character variables to factors
df[, character_vars] <- lapply(df[, character_vars], as.factor)

# # Save the edited dataframe 
# saveRDS(df, file = "df_adult_edited.rds")
# df <- readRDS("df_adult_edited.rds")

## Define train and test data indices
train <- 1:nrow(df_train) 
# Add the number of rows in the test data to each row number of the training data in order to generate and index for the rows of the test data frame.
test <- (1:nrow(df_test))+ nrow(df_train) 

# Create train and test datasets
df_train <- df[train, ]
df_test <- df[test, ]

# Analysis

############   Logistic regression   ############

### Models
## Model 1 (formula 1 = f1): include all variables
# R by default assigns the higher factor level as the positive class
glmf1 <- income ~ .
fit_glmf1 <- glm(glmf1, data = df_train, family = "binomial")

# The summary shows that some dummy variables have missing estimates.
# Workclass does not seem to have a significant influence on the prediction model
summary(fit_glmf1)

# Use the function alias to see if there may be collinearities in the data. It looks like there are none.
alias(fit_glmf1)

# The variance inflation factor of education_num is very high. I will remove education_num for subsequent analyses
vif(fit_glmf1)

# Predict the probabilities using the logistic regression model
probs_glmf1 <- predict(fit_glmf1, newdata = df_test, type = "response")

# Predict the class based on the calculated probabilities
# Set predicted class to ">50K" for observations with probabilities greater than 0.5
pred_glmf1 <- rep("0", 16281)
pred_glmf1[probs_glmf1 > 0.5] <- "1" 
table(pred_glmf1, df_test$income)

contrasts(df$income)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glmf1 <- confusionMatrix(table(pred_glmf1, df_test$income), positive = "1")


## Model 2
# Based on AUC and precision, this model is the best of the logistic regression models
glmf2 <- income ~ . -education_num 
fit_glmf2 <- glm(glmf2, data = df_train, family = "binomial")
summary(fit_glmf2)

# Looking at the variance inflation factor, it looks like there is no more problematic multicollinearity, since none of the adjusted VIF factors are above 5. Although there could be moderate multicollinearity between the variables relationship, marital_status and sex.
vif(fit_glmf2)

# Take a look at independence between workclass and occupation variables.
# It could be worth dropping one of these variables, as the Chi-square test shows significant association between them.
chisq.test(table(df_train$workclass, df_train$occupation))

# Predict the probabilities using the logistic regression model
probs_glmf2 <- predict(fit_glmf2, newdata = df_test, type = "response")

# Predict the class based on the calculated probabilities
# Set predicted class to ">50K" for observations with probabilities greater than 0.5
pred_glmf2 <- rep("0", 16281)
pred_glmf2[probs_glmf2 > 0.5] <- "1" 
table(pred_glmf2, df_test$income)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glmf2 <- confusionMatrix(table(pred_glmf2, df_test$income), positive = "1")


## Model 3
glmf3 <- income ~ . -education_num -workclass 
fit_glmf3 <- glm(glmf3, data = df_train, family = "binomial")
summary(fit_glmf3)
vif(fit_glmf3)

# Predict the probabilities using the logistic regression model
probs_glmf3 <- predict(fit_glmf3, newdata = df_test, type = "response")

# Predict the class based on the calculated probabilities
# Set predicted class to ">50K" for observations with probabilities greater than 0.5
pred_glmf3 <- rep("0", 16281)
pred_glmf3[probs_glmf3 > 0.5] <- "1" 
table(pred_glmf3, df_test$income)

## Create a confusion matrix to determine accuracy, sensitivity etc.
confmatrix_glmf3 <- confusionMatrix(table(pred_glmf3, df_test$income), positive = "1")

specificity_glmf3 <- confmatrix_glmf3$byClass["Specificity"]
accuracy_glmf3 <- as.numeric(confmatrix_glmf3$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_glmf3 <- round(as.numeric(confmatrix_glmf3$byClass["Pos Pred Value"]), 3)

# Draw an ROC curve and extract area under the curve
y <- as.numeric(df_test$income)-1
roc_glmf1 <- roc(y, probs_glmf1)
auc_glmf1 <- roc_glmf1$auc

roc_glmf2 <- roc(y, probs_glmf2)
auc_glmf2 <- roc_glmf2$auc

roc_glmf3 <- roc(y, probs_glmf3)
auc_glmf3 <- roc_glmf3$auc

print(c(auc_glmf1, auc_glmf2, auc_glmf3))

print(c(confmatrix_glmf1$byClass["Pos Pred Value"], confmatrix_glmf2$byClass["Pos Pred Value"], confmatrix_glmf3$byClass["Pos Pred Value"]))

# Use the prediction function from the ROCR package to create a prediction object. This will be used later to create an ROC graph
pred_object_glmf2 <- prediction(probs_glmf2, df_test$income)

# Create a performance object for the ROC curve
perf_glmf2 <- performance(pred_object_glmf2, "tpr", "fpr")  # tpr = true positive rate, fpr = false positive rate

# Calculate the AUC (Area Under the Curve)
auc_glmf2 <- performance(pred_object_glmf2, measure = "auc")@y.values[[1]]


# Compare other statistical methods using this model


##################   Ridge regression   ##################
# Create a model matrix using the model.matrix() function, which converts factors into dummy variables and represent all predictor variables in a matrix format.
# Remove the intercept, because model.matrix() by default includes an intercept column.
matrix_train <- model.matrix(income ~ . -1 -education_num, data = df_train)
y_train <- df_train$income
matrix_test <- model.matrix(income ~ . -1 -education_num, data = df_test)
y_test <- df_test$income

# Fit Ridge regression (alpha = 0 means Ridge in glmnet)
fit_ridge <- glmnet(matrix_train, y = y_train, family = "binomial", alpha = 0)

# Print a summary of the model
print(fit_ridge)

# Use cross-validation to find the best lambda (regularization parameter)
cv_ridge <- cv.glmnet(matrix_train, y_train, family = "binomial", alpha = 0)

# Plot the cross-validation results
plot(cv_ridge)

# Find the best lambda that minimizes the cross-validated error
best_lambda <- cv_ridge$lambda.min

# Refit the model using the best lambda
fit_ridge_best <- glmnet(matrix_train, y_train, family = "binomial", alpha = 0, lambda = best_lambda)

# Coefficients of the final model
coef(fit_ridge_best)

# Make predictions on the test set
probs_ridge_best <- predict(fit_ridge_best, matrix_test, type = "response")

# Convert probabilities to class predictions (optional)
pred_ridge <- ifelse(probs_ridge_best > 0.5, 1, 0)

## Create a confusion matrix to evaluate peformance
confmatrix_ridge <- confusionMatrix(table(pred_ridge, y_test), positive = "1")

specificity_ridge <- confmatrix_ridge$byClass["Specificity"]
accuracy_ridge <- as.numeric(confmatrix_ridge$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_ridge <- round(as.numeric(confmatrix_ridge$byClass["Pos Pred Value"]), 3)

# Extract the coefficients for the best lambda
best_lambda <- cv_ridge$lambda.min
coefficients <- coef(fit_ridge_best, s = best_lambda)

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

###############   XGBoost   ###############

# XGBoost requires numerical inputs so I need to preprocess the data.

# # Convert income variable to binary outcome
#   df$income <- ifelse(df$income == "<=50K", 0, 1)
#   table(df$income)
  
### XGBoost
## Create model matrix for xgboost and define the reponse variable
# One-hot encode categorical variables - i.e. each category for each categorical variable becomes a binary variable coded 0/1
# The model.matrix() function automatically detects and encodes all categorical variables.
# The intercept is removed from the model matrix: "-1"

  matrix_train <- model.matrix(income ~ . -1 -education_num, data = df_train)
  y_train_label <- as.numeric(df_train$income)-1
  matrix_test <- model.matrix(income ~ . -1 -education_num, data = df_test)
  y_test_label <- as.numeric(df_test$income)-1

  # Convert the data to DMatrix, which is XGBoost's optimized data structure
  train_xgb <- xgb.DMatrix(data = matrix_train, label = y_train_label)
  test_xgb <- xgb.DMatrix(data = matrix_test, label = y_test_label)
  
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
  confmatrix_xgb1 <- confusionMatrix(factor(pred_xgb1), factor(y_test_label), positive = "1")
  sensitivity_xgb1 <- confmatrix_xgb1$byClass["Sensitivity"] # Recap: sensitivity is the recall for the positive class
  specificity_xgb1 <- confmatrix_xgb1$byClass["Specificity"] # Recap: specificity is the recall for the negative class
  
  # Extract precision from the confusion matrix.
  # Precision = TP/(TP + FP)
  # Precision is given as "Pos Pred Value" in the caret package
  precision_xgb1 <- round(as.numeric(confmatrix_xgb1$byClass["Pos Pred Value"]), 3)
  
  # Use the prediction function from the ROCR package to create a prediction object. This will be used later to create an ROC graph
  pred_object_xgb <- prediction(probs_xgb1, y_test_label)
  
  # Create a performance object for the ROC curve
  perf_xgb <- performance(pred_object_xgb, "tpr", "fpr")  # tpr = true positive rate, fpr = false positive rate
  
  # Plot the ROC curve
  plot(perf_xgb, col = "red", main = "ROC Curve for XGBoost", lwd = 2)
  
  # Calculate the AUC (Area Under the Curve)
  auc_xgb <- performance(pred_object_xgb, measure = "auc")@y.values[[1]]
  
  # # To get 95% confidence interval on precision:
  # # Extract the table from the confusionMatrix() results
  # cm_matrix_xgb1 <- confmatrix_xgb1$table
  # 
  # # Extract the number of True Positives (TP) and False Positives (FP)
  # TP <- cm_matrix_xgb1[2, 2]  # True Positives (predicted positive and actual positive)
  # FP <- cm_matrix_xgb1[2, 1]  # False Positives (predicted positive and actual negative)
  # 
  # # Calculate precision (TP / (TP + FP))
  # precision_value <- TP / (TP + FP)
  # 
  # # Perform a binomial test to get the confidence interval
  # precision_ci_xgb1 <- binom.test(TP, TP + FP)$conf.int
  # 
  # # Print precision and its confidence interval
  # print(paste("Precision:", round(precision_value, 3)))
  # print(paste("95% Confidence Interval for Precision:", round(precision_ci_xgb1[1], 3), "-", round(precision_ci_xgb1[2], 3)))
  
  # Draw the ROC curve and extract area under the curve
  y <- y_test_label
  roc_xgb1 <- roc(y, probs_xgb1)
  auc_xgb1 <- roc_xgb1$auc
  plot(roc_xgb1)
 
##################    Support Vector Classifier      ##################
  
  # The support vector classifier (SVC) is a method that constructs a set of hyperplanes that separates the training data into two classes. Collinearity can degrade the performance of the classifier when using a linear kernel. To look at the influence of redundant variables in the model, I will run an SVC model with and without the redundant variables using a linear kernel and a support vector machine model with a radial kernel where all variables are included in the model.
  # 
  # I will also perform regularization by playing around with the C tuning parameter, which allows changing the number and severity of the violations to the margin. A smaller C value increases the margin size by allowing more violations, while a larger C value reduces the margin and penalizes misclassifications more severely.
  
  # <!-- How the C (cost) tuning parameter works: -->
  #   
  #   <!-- C Tuning Parameter: The C parameter in SVC controls the degree of regularization. -->
  #   
  #   <!-- A small C (more regularization) allows a larger margin with more violations (misclassified points), making the classifier more tolerant to errors, which can help generalize better. -->
  #   
  #   <!-- A large C (less regularization) reduces the margin and penalizes violations more severely, making the classifier focus on correctly classifying all points, which may lead to overfitting. -->
  #   

  
    ##### SVC with different values of C #####
# Train the SVC model and include the calculation of probabilities
# Train the SVC model at different values of the C tuning parameter
# I use the built-in tune() function in the e1071 library
  # Train an SVM model with a linear kernel - corresponds to support vector classifier (SVC)
  
  # Remove education_num from the dataset before fitting the model
  df_train_svc <- df_train[, !colnames(df_train) %in% c("education_num")]
  df_test_svc <- df_test[, !colnames(df_test) %in% c("education_num")]
  
svcf1 <- income ~.
set.seed(123)

# OBS OBS OBS CHANGED df_train to df_train_svc
  tune_svc <- tune(svm, svcf1, data = df_train_svc, kernel = "linear",
                    ranges = list(cost = c(0.001, 0.1, 1, 10)))

  tune_svc_new <- tune_svc
  saveRDS(tune_svc_new, file = "posts/classification_adult_data/tune_svc_new.rds")
# This computation takes quite some time, so I am saving the results.
# saveRDS(tune_svc, file = "posts/classification_adult_data/tune_svc.rds")
  tune_svc <- readRDS("posts/classification_adult_data/tune_svc_new.rds")
  
  # Access the cross-validation errors for each of the models
  summary(tune_svc)
  
  # Extract the best model
  bestmod_svc <- tune_svc$best.model
  
  ## Predict the class and probabilities using the best model
  pred_svc <- predict(bestmod_svc, df_test, decision.values = TRUE)
  
  ## Evaluate the model's performance
  confmatrix_svc <- confusionMatrix(pred_svc, df_test$income, positive = "1")
  
  # Extract accuracy
  accuracy_svc <- as.numeric(confmatrix_svc$overall["Accuracy"])
  
  # Extract precision from the confusion matrix.
  # Precision = TP/(TP + FP)
  # Precision is given as "Pos Pred Value" in the caret package
  precision_svc <- round(as.numeric(confmatrix_svc$byClass["Pos Pred Value"]), 3)
  
  ## In order to prepare and ROC curve
  # Extract the decision values (distance from the decision boundary)
  decision_values_svc <- attributes(pred_svc)$decision.values
  
  # Use ROCR to plot the ROC curve
  # Create a prediction object using decision values and true labels
  pred_object_svc <- prediction(-decision_values_svc, y_test_label)
  
  # Create a performance object for the ROC curve
  perf_svc <- performance(pred_object_svc, "tpr", "fpr")
  
  # Plot the ROC curve
  plot(perf_svc, col = "green", main = "ROC Curve for Support Vector Classifier", lwd = 2)
  
  # Calculate the AUC (Area Under the Curve)
  auc_svc <- performance(pred_object_svc, measure = "auc")@y.values[[1]]
  
# Variable importance
  
# Calculate the linear SVM model's coefficients by multiplying the transpose of the support vector coefficients by the support vectors
  svc_coefficients <- t(bestmod_svc$coefs) %*% bestmod_svc$SV
  
  # Extract the variable names (second element of dimnames)
  variable_names <- dimnames(svc_coefficients)[[2]]
  
  # Extract the coefficients and the names
  svc_coeff_values <- as.vector(svc_coefficients)  # Extract the numeric values
  variable_names <- dimnames(svc_coefficients)[[2]]
  
  # Combine the names and coefficients into a named vector
  # Remember that in this case:
  # Positive coefficients (in the output from my SVM model) will push the decision toward the negative class.
  # Negative coefficients will push the decision toward the positive class.
  svc_coefficients_named <- setNames(-svc_coeff_values, svc_coeff_names)
  
  # Sort the coefficients by their absolute values, keeping the names intact
  # Select the top 20 largest estimates (based on absolute values)
    svc_coefficients_ordered <- svc_coefficients_named[order(abs(svc_coefficients_named), decreasing = TRUE)]
    importance_svc_top20 <- svc_coefficients_ordered[1:20]
    
    # Convert the named vector of top 20 into a dataframe for plotting
    importance_svc_top20 <- data.frame(
      Variable = names(importance_svc_top20),
      Coefficient = importance_svc_top20
    )
    
    # Create a bar plot with the actual values (not the absolute ones)
    ggplot(importance_svc_top20, aes(x = reorder(Variable, abs(Coefficient)), y = Coefficient)) +
      geom_bar(stat = "identity", aes(fill = ifelse(Coefficient > 0, "Positive", "Negative"))) +
      coord_flip() +
      labs(title = "Support Vector Classifier Variable Importance", 
           x = "Variables", 
           y = "Coefficient Value") +
      theme_minimal() +
      scale_fill_manual(values = c("Negative" = "darkorange", "Positive" = "steelblue"), guide = "none")
   
    ##### SVM (radial kernel) #####

  # Train the SVM model at different values of the C tuning parameter
  # I use the built-in tune() function in the e1071 library
  # Train an SVM model with a linear kernel - corresponds to support vector classifier (SVC)  
  svmf1 <- income ~. 
  
  set.seed(123)
  tune_svm <- tune(svm, svmf1, data = df_train_svc,
                  kernel = "radial",
                  ranges = list(
                     cost = c(0.001, 0.1, 1),
                    gamma = c(0.5, 1, 2)))
  
  # The cost = 10 gave a bigger error
  # tune_svm2 <- tune(svm, svmf1, data = df_train_svc,
  #                   kernel = "radial",
  #                   ranges = list(
  #                     cost = c(1, 10),
  #                     gamma = 0.5
  #                     )
  #                   )
  
  # This computation takes quite some time, so I am saving the results.
  saveRDS(tune_svm, file = "posts/classification_adult_data/tune_svm.rds")
  tune_svm <- readRDS("posts/classification_adult_data/tune_svm.rds")
  
  summary(tune_svm)
  

  
###############        Results           ###############

plot(perf_svc, col = "green", main = "ROC Curve for SVM (Linear Kernel)", lwd = 2)
plot(perf_xgb, col = "red", lwd = 2, add = TRUE)
plot(perf_glmf2, col = "blue", lwd = 2, add = TRUE)

# Add a legend to the plot with AUC values
legend("bottomright",
       legend = c(paste("SVM AUC:", round(auc_svc, 3)),
                  paste("XGBoost AUC:", round(auc_xgb, 3)),
                  paste("Logistic Regression AUC:", round(auc_glm, 3))),
       col = c("green", "red", "blue"),
       lwd = 2)



importance_plot_glm <- ggplot(importance_glmf2_top20, aes(x = reorder(Variable, abs(Estimate)), y = Estimate)) +
  geom_bar(stat = "identity", aes(fill = ifelse(Estimate > 0, "Positive", "Negative"))) +
  coord_flip() +  
  labs(title = "Logistic Regression Top 20 Important Variables",
       x = "Variables",
       y = "Coefficient Estimates") +
  theme_minimal() +
  # Define custom colors for positive (blue) and negative (orange)
  scale_fill_manual(values = c("Negative" = "darkorange", "Positive" = "steelblue"), guide = "none")


###### SVC without cross validation, default method where cost = 1  ######    
# Train an SVM model with a linear kernel - corresponds to support vector classifier (SVC)
# I don't use probability estimation - it is computationally too expensive.
svcf1 <- income ~. -education_num
fit_svc1 <- svm(svcf1, data = df_train, kernel = "linear", decision.values = TRUE)

# Predict using the SVC model and get decision values
# Predict on the test set with decision values (not probabilities)
pred_svc1 <- predict(fit_svc1, df_test, decision.values = TRUE)

# Prepare confusion matrix
table(pred_svc1, df_test$income)
confmatrix_svc1 <- confusionMatrix(table(pred_svc1, df_test$income), positive = "1")
# Extract accuracy
accuracy_svc1 <- as.numeric(confmatrix_svc1$overall["Accuracy"])

# Extract precision from the confusion matrix.
# Precision = TP/(TP + FP)
# Precision is given as "Pos Pred Value" in the caret package
precision_svc1 <- round(as.numeric(confmatrix_svc1$byClass["Pos Pred Value"]), 3)


## In order to prepare and ROC curve
# Extract the decision values (distance from the decision boundary)
decision_values <- attributes(pred_svc1)$decision.values

# Use ROCR to plot the ROC curve
# Create a prediction object using decision values and true labels
pred_object_svc <- prediction(-decision_values, y_test_label)

# Create a performance object for the ROC curve
perf_svc <- performance(pred_object_svc, "tpr", "fpr")

# Plot the ROC curve
plot(perf_svc, col = "green", main = "ROC Curve for SVM (Linear Kernel)", lwd = 2)

# Calculate the AUC (Area Under the Curve)
auc_svc <- performance(pred_object_svc, measure = "auc")@y.values[[1]]
cat("AUC:", auc_svc, "\n")




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

####################      Script not used      ##################
    
    # Do a Pearson's Chi-squared test to determine the independence between the "workclass" and "occupation" variables.
    # The test shows that there is a strong association (significant p-value and large Chi-square statistic) between these two variables.
    chisq.test(table(df_train$workclass, df_train$occupation))
    
    
    # Take a look at independence between marital_status and relationship variables.
    # It could be worth dropping one of these variables, as the Chi-square test shows significant association between them.
    chisq.test(table(df_train$marital_status, df_train$relationship), simulate.p.value = TRUE)
    