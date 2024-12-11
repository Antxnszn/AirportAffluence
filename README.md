# NotACrowd v.1.0

For airports passengers.

## Description

This project aims to predict passenger traffic at airports based on historical data, specifically focusing on the month and year. By using machine learning techniques such as KMeans clustering and Random Forest, the model forecasts the number of passengers for a given date, helping airports prepare for crowd management and optimize resources.

### Installing
Before run this project, you have to install Anaconda Distribution 
Link: [https://www.anaconda.com/download]

### Dependencies

* pandas >= 1.0.0
* scikit-learn >= 0.24.0
* numpy >= 1.18.0
* matplotlib (for data visualization, optional but useful)
* seaborn (for better visualizations, optional)
* datetime (Python built-in library, no installation needed)

###Recomendation
* Create a virtual environment in order to prevent any version conflicts with another dependencies that you had previously install.
  
```
conda create -n [VIRTUAL ENV NAME]
```
```
conda activate [VIRTUAL ENV NAME]
```
When you have your virtual environment active, install dependencies
As this examples for all the dependencies:
```
pip install pandas
```

### Executing program
When all the requeriments were satisfied, run the project
```
python model.py
```


