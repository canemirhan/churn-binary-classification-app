# Importing necessary libraries
import pandas as pd
import _pickle as cPickle
import streamlit as st

from utils import FeatureEngineeringUtil

# Loading base dataframe and model
df = pd.read_csv("before_fteng_data.csv",index_col=0)
model = cPickle.load(open("xgbc_model.pkl","rb"))

def prediction(base_df:pd.DataFrame,user:pd.DataFrame,model):
    """
    Predicts whether customer is exited or not. Gives the result and its probability at the end.

    Parameters
    ----------
    base_df: pd.DataFrame
        Base dataframe
    user: pd.DataFrame
        User dataframe
    model

    """
    df_user = pd.concat([base_df,user],axis=0)
    x,_ = FeatureEngineeringUtil().churn_prep(df_user)
    user_x = x.iloc[-1:,:]

    result = model.predict(user_x)

    if result[0] == 1:
        st.error(f"Exited by {round(model.predict_proba(user_x)[:,1][0] * 100, 3)}% chance...")
    else:
        st.success(f"Not Exited by {round(model.predict_proba(user_x)[:,0][0] * 100, 3)}% chance...")

## STREAMLIT PART ##
st.title("Customer Churn Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    cr_score = st.slider("Credit score",300,850,650)

    age = st.slider("Age", 18, 120, 35)


with col2:
    tenure = st.number_input("Enter your tenure")

    balance = st.number_input("Enter your balance")

    salary = st.number_input("Enter your salary (Estimated or Exact)")

    num_prod = st.selectbox("Select the number of products you have", [1, 2, 3, 4])


with col3:
    cr_card = st.radio("Do you have credit card", ["Yes", "No"])
    cr_card = [1 if cr_card == "Yes" else 0][0]

    act_mem = st.radio("Are you an active member", ["Yes", "No"])
    act_mem = [1 if act_mem == "Yes" else 0][0]

    geo = st.radio("Select your country", ["France","Germany","Spain"])

    sex = st.radio("Select your gender", ["Male","Female"])


# Creating user dataframe
user =pd.DataFrame({'creditscore':[cr_score],
                    'geography':[geo],
                    'gender':[sex],
                    'age':[age],
                    'tenure':[tenure],
                    'balance':[balance],
                    'numofproducts':[num_prod],
                    'hascrcard':[cr_card],
                    'isactivemember':[act_mem],
                    'estimatedsalary':[salary],
                    'exited':[-1]})


# Prediction and the output
if col3.button("Predict",use_container_width=True):
    prediction(df,user,model)






