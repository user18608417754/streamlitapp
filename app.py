import streamlit as st
import pandas as pd
import joblib
import numpy as np

title = "Prediction Models for Neurosyphilis Diagnosis"

header_style = """
    text-align: center;
    font-size: 28px;
    border-bottom: 1px solid black;
    margin-bottom: 15px;
    font-weight: bold;
"""

st.set_page_config(
    page_title=f"{title}",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<div style='{header_style}'>{title}</div>", unsafe_allow_html=True)

tabs = st.tabs(["🚀 Model 1", "🚀 Model 2"])

st.markdown("""
    <style>
        [role="tablist"] {
            justify-content: center;
        }
        [role="tablist"] button p {
            font-size: 20px;
        }
    </style>""", unsafe_allow_html=True)

t = """<div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
Overall</div><div>
🔸 Welcome to the Shiny App for Neurosyphilis Diagnosis.<br>
🔸 You can use this to predict the risk of developing a neurosyphilis outcome in syphilis patients without HIV infection.<br>
🔸 This page exhibits Model 1. Model 1 uses CSF-NTT as predictive outcome.</div>"""
t1 = """<div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
Overall</div><div>
🔸 Welcome to the Shiny App for Neurosyphilis Diagnosis.<br>
🔸 You can use this to predict the risk of developing a neurosyphilis outcome in syphilis patients without HIV infection.<br>
🔸 This page exhibits Model 2. Model 2 uses all_scores as predictive outcome.</div>"""

sex = {"Female":0, "Male":1}
v1 = {'Without neurological symptoms':0,'With neurological symptoms':1}
v2 = {'Negative':0, 'Postive':1}

c1 = ['Age','CSF_WBC', 'CSF_TT']
c2 = ['CSF_TT', 'NS_Symptom']

model1 = joblib.load("model2.pkl")
scaler1 = joblib.load("scaler2.pkl")
model2 = joblib.load("model4.pkl")
scaler2 = joblib.load("scaler4.pkl")

with tabs[0]:
    col = st.columns(3)
    col[0].markdown(t, unsafe_allow_html=True)
    
    d1 = {}
    
    col[1].markdown('''
    <div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
    Predict input
    </div>''', unsafe_allow_html=True)
    
    col[1].markdown('''
    <div style="text-align: left; font-weight: bold; color: black; margin-bottom: 8px;">
    1. Demographic Characteristics
    </div>''', unsafe_allow_html=True)
    
    d1["Age"] = col[1].number_input("**Age(years):**", value=50, min_value=0, max_value=100, step=1)
    
    col[1].markdown('''
    <div style="text-align: left; font-weight: bold; color: black; margin-bottom: 8px;">
    2. Laboratory testing
    </div>''', unsafe_allow_html=True)
    d1["CSF_WBC"] = col[1].number_input("**CSF_WBC(cells/ul):**", value=2, min_value=0, step=1)
    d1["CSF_TT"] = v2[col[1].selectbox("**CSF_TT:**", v2)]
    
    #col[1].markdown('''<div style="color: black; border-radius: 8px; padding: 8px; border: 1px solid black;">This page exhibits Model 2. Model 2 uses all_scores as predictive outcome.</div>''', unsafe_allow_html=True)
    
    col[2].markdown('''
    <div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
    Predict result
    </div>''', unsafe_allow_html=True)
    col[2].markdown('''<div style="color: black; border: 1px solid black; border-radius: 8px; padding: 8px; margin-bottom: 8px;">Risk defined as the probability specifying the likelihood of classification to neurosyphilis for each syphilis patient. Diagnosis assessed at the threshold probability of 50%.</div>''', unsafe_allow_html=True)
    
    pre1 = pd.DataFrame(np.array([list(d1.values())]))
    pre1.columns = c1
    pre1 = model1.predict_proba(scaler1.transform(pre1))
    
    if pre1[0][1]>0.5:
        s1 = "color: red;"
        r1 = f"High risk prediction, {round(float(pre1[0][1])*100, 3)}%"
    else:
        s1 = "color: green;"
        r1 = f"Low risk prediction, {round(float(pre1[0][1])*100, 3)}%"
    
    col[2].markdown(f'''<div style="color: black; border: 1px solid black; border-radius: 8px; padding: 8px;"><div style="border-bottom: 1px solid black; font-weight: bold;">Risk Prediction and Final Diagnosis</div><span style="{s1}">{r1}</span></div>''', unsafe_allow_html=True)
    
with tabs[1]:
    col = st.columns(3)
    col[0].markdown(t1, unsafe_allow_html=True)
    
    d2 = {}
    
    col[1].markdown('''
    <div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
    Predict input
    </div>''', unsafe_allow_html=True)
    
    col[1].markdown('''
    <div style="text-align: left; font-weight: bold; color: black; margin-bottom: 8px;">
    1. Clinical Features
    </div>''', unsafe_allow_html=True)
    d2["NS_Symptom"] = v1[col[1].selectbox("**NS_Symptom:**", v1, key="Neurological2")]
    
    col[1].markdown('''
    <div style="text-align: left; font-weight: bold; color: black; margin-bottom: 8px;">
    2. Laboratory testing
    </div>''', unsafe_allow_html=True)

    d2["CSF_TT"] = v2[col[1].selectbox("**CSF_TT:**", v2, key="Serum2")]
    
    #col[1].markdown('''<div style="color: black; border-radius: 8px; padding: 8px; border: 1px solid black;">This page exhibits Model 2. Model 2 uses all_scores as predictive outcome.</div>''', unsafe_allow_html=True)
    
    col[2].markdown('''
    <div style="font-size: 20px; text-align: left; font-weight: bold; color: black; border-bottom: 1px solid black; margin-bottom: 8px;">
    Predict result
    </div>''', unsafe_allow_html=True)
    col[2].markdown('''<div style="color: black; border: 1px solid black; border-radius: 8px; padding: 8px; margin-bottom: 8px;">Risk defined as the probability specifying the likelihood of classification to neurosyphilis for each syphilis patient. Diagnosis assessed at the threshold probability of 50%.</div>''', unsafe_allow_html=True)
    
    pre2 = pd.DataFrame(np.array([list(d2.values())]))
    pre2.columns = c2
    pre2 = model2.predict_proba(scaler2.transform(pre2))
    
    if pre2[0][1]>0.5:
        s2 = "color: red;"
        r2 = f"High risk prediction, {round(float(pre2[0][1])*100, 3)}%"
    else:
        s2 = "color: green;"
        r2 = f"Low risk prediction, {round(float(pre2[0][1])*100, 3)}%"
    
    col[2].markdown(f'''<div style="color: black; border: 1px solid black; border-radius: 8px; padding: 8px;"><div style="border-bottom: 1px solid black; font-weight: bold;">Risk Prediction and Final Diagnosis</div><span style="{s2}">{r2}</span></div>''', unsafe_allow_html=True)
