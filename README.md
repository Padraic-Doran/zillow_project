# zillow_project
House Price Regression Project
Objective
Use property data to predict the values of single unit properties for properties sold in May and June of 2017.

Goals
Your customer is the zillow data science team. state your goals as if you were delivering this to zillow. They have asked for something from you and you are basically communicating in a more concise way, and very clearly, the goals as you understand them and as you have taken and acted upon through your research.

Deliverables
What should the zillow team expect to receive from you? Again, as you were communicating to them, not to your instructors.

1. A report (in the form of a presentation, both verbal and through a slides)
The report/presentation slides should summarize your findings about the drivers of the Zestimate error. This will come from the analysis you do during the exploration phase of the pipeline. In the report, you will have charts that visually tell the story of what is driving the errors.

2. A github repository containing your jupyter notebook that walks through the pipeline along with the .py files necessary to reproduce your model.
model must be reproducible by someone with their own env.py file

The Pipeline
PROJECT PLANNING & README
Brainstorming ideas, hypotheses, related to how variables might impact or relate to each other, both within independent variables and between the independent variables and dependent variable, and also related to any ideas for new features you may have while first looking at the existing variables and challenge ahead of you.

Have a detailed README.md file for anyone who wants to check out your project. In this file should be a description of what the project is, and any instructions necessary for someone else to clone your project and run the code on their own laptop.

ACQUIRE:
Goal: leave this section with a dataframe ready to prepare.

The ad hoc part includes summarizing your data as you read it in and begin to explore, look at the first few rows, data types, summary stats, column names, shape of the data frame, etc.

acquire.py: The reproducible part is the gathering data from SQL.

PREP:
Goal: leave this section with a dataset that is ready to be analyzed. Data types are appropriate, missing values have been addressed, as have any data integrity issues.

The ad hoc part includes plotting the distributions of individual variables and using those plots to identify outliers and if those should be handled (and if so, how), identify unit scales to identify how to best scale the numeric data, as well as finding erroneous or invalid data that may exist in your dataframe.

Add a data dictionary in your notebook that defines all fields used in either your model or your analysis, and answers the question: why did you use the fields you used, e.g. why did you use bedroom_field1 over bedroom_field2, not why did you use number of bedrooms!

prep.py: The reproducible part is the handling of missing values, fixing data integrity issues, changing data types, etc.

SPLIT & SCALE:
Goal: leave this section with 2 dataframes (train & test) and scaled data

split_scale.py: split data, scale data (create scaler, fit the scaler to train and transform the train and test using that scaler)

DATA EXPLORATION
Goal: Address each of the questions you posed in your planning and brainstorming and any others you have come up with along the way through visual or statistical analysis.

When you have completed this step, you will have the findings from your analysis that will be used in your final report, answers to specific questions your customers has asked, and information to move forward toward building a model.

Run at least 1 t-test and 1 correlation test (but as many as you need!)

Visualize all combinations of variables in some way(s).

What independent variables are correlated with the dependent? (this is good)

Which independent variables are correlated with other independent variables? (this is not so good and needs to be addressed)

Summarize your takeaways and conclusions.

FEATURE SELECTION
Goal: leave this section with a dataframe with the features to be used to build your model.

Are there new features you could create based on existing features that might be helpful?

You could use feature selection techniques to see if there are any that are not adding value to the model.

feature_selection.py: to run whatever functions need to be run to end with a dataframe that contains the features that will be used to model the data.

MODELING & EVALUATION
Goal: develop a regression model that performs better than a baseline.

You must evaluate a baseline model, and show how the model you end up with performs better than that.

model.py: will have the functions to fit, predict and evaluate the model

Your notebook will contain various algorithms and/or hyperparameters tried, along with the evaluation code and results, before settling on the final algorithm.

Be sure and evaluate your model using the standard techniques: plotting the residuals, computing the evaluation metric (SSE, RMSE, and/or MSE), comparing to baseline, plotting 
y
 by 
^
y