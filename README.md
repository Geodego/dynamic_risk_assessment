# Dynamic Risk Assessment System

Udacity Machine Learning DevOps Engineer project.

## Project Overview

### Background

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is 
extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease 
the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them 
not to exit their contracts. However, the client management team is small, and they're not able to stay in close 
contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk 
of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client
managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, 
and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to 
set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and 
scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments 
that are as accurate as possible and minimize client attrition.

### Project Steps Overview
You'll complete the project by proceeding through 5 steps:

- **Data ingestion**. Automatically check a database for new data that can be used for model training. Compile all 
training data to a training dataset and save it to persistent storage. Write metrics related to the completed data 
ingestion tasks to persistent storage.
- **Training, scoring, and deploying**. Write scripts that train an ML model that predicts attrition risk, and score the 
model. Write the model and the scoring metrics to persistent storage.
- **Diagnostics**. Determine and save summary statistics related to a dataset. Time the performance of model training and 
scoring scripts. Check for dependency changes and package updates.
- **Reporting**. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that 
can return model predictions and metrics.
- **Process Automation**. Create a script and cron job that automatically run all previous steps at regular intervals.

## Data Ingestion
Data ingestion is important because all ML models require datasets for training. Instead of using a single, static 
dataset, you're going to create a script that's flexible enough to work with constantly changing sets of input files. 
This step will make your data ingestion go smoothly and easily, even if the data itself is complex.

In this step, you'll read data files into Python, and write them to an output file that will be your master dataset. 
You'll also save a record of the files you've read.

When we're initially setting up the project, our config.json file will be set to read ```practicedata``` and write 
```practicemodels```. When we're ready to finish the project, you will need to change the locations specified in 
```config.json``` so that we're reading our actual, ```sourcedata``` and we're writing to our ```models``` directory.

### Reading Data and Compiling a Dataset
In the first part of your data ingestion.py script, you'll read a collection of csv files into Python.

The location of the csv files you'll be working with is specified in the config.json starter file, in an entry called 
`input_folder_path`. In your starter version of config.py, this entry's value is set to the `/practicedata/` directory.

You need to add code to your `ingestion.py` starter file so that it can automatically detect all of the csv files in 
the directory specified in the `input_folder_path`. Each of the files in the `input_folder_path` represents a 
different dataset. You'll need to combine the data in all of these individual datasets into a single pandas DataFrame.

You shouldn't manually write file names in your script: your script needs to automatically detect every file name in 
the directory. Your script should work even if we change the number of files or the file names in the 
`input_folder_path`.

It's possible that some of the datasets that you read and combine will contain duplicate rows. So, you should de-dupe 
the single pandas DataFrame you create, and ensure that it only contains unique rows.

### Writing the Dataset
Now that you have a single pandas DataFrame containing all of your data, you need to write that dataset to storage in 
your workspace. You can save it to a file called `finaldata.csv`. Save this file to the directory that's specified in 
the `output_folder_path` entry of your `config.json` configuration file. In your starter version of `config.json`, 
the output_folder_path entry is set to `/ingesteddata/`, so your dataset will be saved to `/ingesteddata/`.

### Saving a record of the ingestion
For later steps in the project, you'll need to have a record of which files you read to create your `finaldata.csv` 
dataset. You need to create a record of all of the files you read in this step, and save the record on your workspace 
as a Python list.

You can store this record in a file called `ingestedfiles.txt`. This file should contain a list of the filenames of 
every .csv you've read in your `ingestion.py` script. You can also save this file to the directory that's specified in 
the `output_folder_path entry` of your `config.json` configuration file.

## Training, Scoring, and Deploying an ML Model
Training and Scoring an ML model is important because ML models are only worth deploying if they've been trained, 
and we're always interested in re-training in the hope that we can improve our model accuracy. Re-training and scoring, 
as we'll do in this step, are crucial so we can get the highest possible model accuracy.

This step will require you to write three scripts. One script will be for training an ML model, another will be for 
generating scoring metrics for the model, and the third will be for deploying the trained model.

The data in finaldata.csv represents records of corporations, their characteristics, and their historical attrition 
records. One row represents a hypothetical corporation. There are five columns in the dataset:
- "corporation", which contains four-character abbreviations for names of corporations
- "lastmonth_activity", which contains the level of activity associated with each corporation over the previous month
- "lastyear_activity", which contains the level of activity associated with each corporation over the previous year
- "number_of_employees", which contains the number of employees who work for the corporation
- "exited", which contains a record of whether the corporation exited their contract (1 indicates that the corporation 
exited, and 0 indicates that the corporation did not exit)

The dataset's final column, "exited", is the target variable for our predictions. The first column, "corporation", 
will not be used in modeling. The other three numeric columns will all be used as predictors in your ML model.

### Model Training
Build a function that accomplishes model training for an attrition risk assessment ML model. Your model training 
function should accomplish the following:
- Read in finaldata.csv using the pandas module. The directory that you read from is specified in the `output_folder_path` 
of your `config.json` starter file.
- Use the scikit-learn module to train an ML model on your data. 
- Write the trained model to your workspace, in a file called trainedmodel.pkl. The directory you'll save it in is 
specified in the `output_model_path` entry of your `config.json` starter file.
You can write code that will accomplish all of these steps in `training.py`, which is included in your starter files.

### Model Scoring
You need to write a function that accomplishes model scoring. You can write this function in the starter file called 
scoring.py. To accomplish model scoring, you need to do the following:
- Read in test data from the directory specified in the test_data_path of your config.json file
- Read in your trained ML model from the directory specified in the output_model_path entry of your config.json file
- Calculate the F1 score of your trained model on your testing data
- Write the F1 score to a file in your workspace called latestscore.txt. You should save this file to the directory 
specified in the output_model_path entry of your config.json file

### Model Deployment
Finally, you need to write a function that will deploy your model. You can write this function in the starter file called deployment.py.

Your model deployment function will not create new files; it will only copy existing files. It will copy your trained 
model (`trainedmodel.pkl`), your model score (`latestscore.txt`), and a record of your ingested data 
(`ingestedfiles.txt`). It will copy all three of these files from their original locations to a production deployment directory. 
The location of the production deployment directory is specified in the `prod_deployment_path` entry of your 
config.json starter file.

## Model and Data Diagnostics
Model and data diagnostics are important because they will help you find problems - if any exist - in your model and 
data. Finding and understanding any problems that might exist will help you resolve the problems quickly and make sure 
that your model performs as well as possible.

In this step, you'll create a script that performs diagnostic tests related to your model as well as your data.

### Model Predictions
You need a function that returns predictions made by your deployed model.

This function should take an argument that consists of a dataset, in a pandas DataFrame format. It should read the 
deployed model from the directory specified in the `prod_deployment_path` entry of your config.json file.

The function uses the deployed model to make predictions for each row of the input dataset. Its output should be a 
list of predictions. This list should have the same length as the number of rows in the input dataset.

### Summary statistics
You also need a function that calculates summary statistics on your data.

The summary statistics you should calculate are means, medians, and standard deviations. You should calculate each of 
these for each numeric column in your data.

This function should calculate these summary statistics for the dataset stored in the directory specified by 
`output_folder_path` `in config.json`. It should output a Python list containing all of the summary statistics for 
every numeric column of the input dataset.

### Missing Data
Next, you should write a function to check for missing data. Your function needs to count the number of NA values in 
each column of your dataset. Then, it needs to calculate what percent of each column consists of NA values.

The function should count missing data for the dataset stored in the directory specified by `output_folder_path` in 
config.json. It will return a list with the same number of elements as the number of columns in your dataset. 
Each element of the list will be the percent of NA values in a particular column of your data.

### Timing
Next, you should create a function that times how long it takes to perform the important tasks of your project. 
The important tasks you need to time are: data ingestion and model training.

This function doesn't need any input arguments. It should return a Python list consisting of two timing measurements 
in seconds: one measurement for data ingestion, and one measurement for model training.

### Dependencies
It's important to make sure that the modules you're importing are up-to-date.

In this step, you'll write a function that checks the current and latest versions of all the modules that your 
scripts use (the current version is recorded in `requirements.txt`). It will output a table with three columns: 
- the first column will show the name of a Python module that you're using 
- the second column will show the currently installed version of that Python module
- the third column will show the most recent available version of that Python module.

To get the best, most authoritative information about Python modules, you should rely on Python's official package 
manager, pip. Your script should run a pip command in your workspace Terminal to get the information you need for 
this step.

**Note**: Dependencies don’t need to be re-installed or changed, since this is just a check.

## Model Reporting
Model reporting is important because reporting allows us as data scientists to be aware of all aspects of our data, 
our model, and our training processes, as well as their performance. Also, automated reporting enables us to keep 
stakeholders and leaders quickly and reliably informed about our ML efforts.

In this step, you'll write scripts that create reports related to your ML model, its performance, and related 
diagnostics.

### Generating Plots
You need to update the **reporting.py** script so that it generates plots related to your ML model's performance.

In order to generate plots, you need to call the model prediction function that you created 
**diagnostics.py** in Step 3. The function will use the test data from the directory specified in the `test_data_path` 
entry of your **config.json** starter file as input dataset. You can use this function to obtain a list of predicted 
values from your model. After you obtain predicted values and actual values for your data, you can use these to 
generate a confusion matrix plot. Your **reporting.py** script should save your confusion matrix plot to a file in your 
workspace called **confusionmatrix.png**. The **confusionmatrix.png** file can be saved in the directory specified 
in the `output_model_path` entry of your **config.json** file.

## API Setup
We set up an API using **app.py** so that we can easily access ML diagnostics and results. The API has four endpoints: 
- one for model predictions: `/prediction`
- one for model scoring: `/scoring`
- one for summary statistics: `/summarystats`
- one for other diagnostics: `/diagnostics`

### Calling the API endpoints
The **apicalls.py** script calls each of the API endpoints, combine the outputs, and write the combined outputs to a 
file called **apireturns.txt**. The **apireturns.txt** file is saved in the directory specified in the 
`output_model_path` entry of the **config.json** file.

## Process automation
Process automation is important because it will eliminate the need for you to manually perform the individual steps of 
the ML model scoring, monitoring, and re-deployment process.

In this step, you'll create scripts that automate the ML model scoring and monitoring process.

This step includes checking for the criteria that will require model re-deployment, and re-deploying models as 
necessary.

The full process that you'll automate is shown in the following figure:
![](images/automation.png)

### Updating config.json
For testing purposes the  **config.json** file contains an entry called input_folder_path with a value equal to 
`/practicedata/`. When testing, all of the training, scoring, and reporting  is accomplished relying on the contents 
of this directory. As the name "practicedata" suggests, this folder's datasets is provided to practice and test the 
scripts. Once all of the scripts have been completed, we want to stop working with practice data and start working with 
production data. Such production data can be found in the folder called `/sourcedata/`.

Changing from practice data to production data only requires changing one thing. We need to change the 
`input_folder_path` entry in the **config.json** file. Instead of `/practicedata/`, we need to change it to be 
`/sourcedata/`. Since all of the scripts read this value from **config.json**, making that one change will enable all 
of the scripts to work with this new, correct data instead of our practice data.

We also need to change the `output_model_path`. In the training version of **config.json**, the value for this entry 
is set to `/practicemodels/`. We should change it to `/models/` for storing production models instead of practice 
models.



