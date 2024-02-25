# Churn Binary Classification Streamlit App

In this project, we are trying to classify customer churn either "exited" or "not exited".

## Table of Contents
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Streamlit Application](#streamlit-application)
    - [Installation](#installation)
  - [Contribution](#contribution)
  - [Acknowledgments](#acknowledgments)

## Dataset
The dataset `data.csv` used for this project was taken from kaggle competition called "Binary Classification with a Bank Churn Dataset". If you want to get more details about the dataset, you can check [this link](https://www.kaggle.com/competitions/playground-series-s4e1/data).


## Project Structure
This project mainly consists of `notebook.ipynb` used for data visualizations and model preparation, `utils.py` included many useful functions for feature engineering, eda etc. and `app.py` for the streamlit app.

## Streamlit Application
You can start to use the app from [this link](https://custchurnbc.streamlit.app/) deployed on Streamlit Cloud. If you want to use the app from your local machine, you can follow `Installation` steps.

### Installation
Clone the repository to your local machine:
    
    git clone https://github.com/canemirhan/churn-binary-classification-app
    
    
Navigate to the project directory:
    
    cd customer-churn-app

    
Install the required dependencies:
    
    pip install -r requirements.txt
    
    
Run the Streamlit app:
    
    streamlit run app.py
    
    
    
Finally, access the application through your web browser at `http://localhost:8501`.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on this project.








