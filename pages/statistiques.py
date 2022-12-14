# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:32:55 2022

@author: mato
"""

import pandas as pd
import base64

import dash
from dash import html, dcc, callback, Input, Output
from dash.dependencies import State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import dash_labs as dl
from dash import dash_table

import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.figure_factory as ff

import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler


# MAchine learning
from sklearn import datasets, ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

# Score of models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import classification_report


#mydata
'''
df_pointés = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\RESULTATS\processing_pointesinterfaces_v1.csv", sep=";")
df_hv = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\HV\HV_processed.csv", sep=";")
df_aem = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\AEM\aem_processed_v1.csv", sep=";")
'''

conn = psycopg2.connect(host="dpg-cecsgdsgqg459grbbo70-a.frankfurt-postgres.render.com", port = 5432, database="dbmayotte", user="ciapy", password="fkid4gbdDC58V5wwjZGA4rqqcLnRVyZ0")

cur = conn.cursor()

cur.execute("select * from  public.dataset")
df = pd.DataFrame(cur.fetchall(), columns=['index','x_epsg4471', 'y_epsg4471', 'rho','z', 'target'])

conn.close()

dataml = df[['x_epsg4471', 'y_epsg4471', 'rho','z', 'target']]
#Fonction

def general_analysis(df):
    global shp_df
    shp_df = df.shape
    global dtype_df
    dtype_df =df.dtypes
    dtype_df= dtype_df.to_frame()
    global info_df
    info_df = df.info()
    global isna_df
    isna_df = df.isna().sum()
    return shp_df,dtype_df,info_df,isna_df

def univarirate_analysis_A(df):
    descr_df= df.describe(include='all')
    for i, column in enumerate(df.columns, 1):
        plt.subplots(i)
        plot_hist = sns.histplot(df[column])
    return descr_df, plot_hist
    
def correlation_matrix(df):
    corr_df= df.corr()
    heatmapcorr_df = sns.heatmap(df.corr(), annot=True, linewidths=0.5)
    return corr_df, heatmapcorr_df


def entrainement_evaluation_du_modele(data, modele_donne, parameters, metrics_GS):
        
    # target preprocessing
    lb_encod = LabelEncoder()
    global y

    y = lb_encod.fit_transform(data['target'])
        
    
    # features preprocessing
    global X

    X = data.drop(columns='target')
    X.head()    

    # Division en groupes de training et d'évaluation
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    

    # Declare model for Grid Search
    model_GS = modele_donne

    # Declare the pipeline
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model_GS)]
        )

    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    
    # Declare the Grid Search method
    grid = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = metrics,
                        refit = metrics_GS, cv = 5, n_jobs =-1, verbose = 1)

    # Fit the model
    grid.fit(X_train, y_train)

    # Evaluate cross validation performance 
    print()
    print("model: ", modele_donne)
    print("CV - Best score:", round(grid.best_score_,3))
    print("CV - best parameters:", grid.best_params_)
    #print("CV - best estimator :", grid.best_params_)
    
    # cv_results_['mean_fit_time'] donne un array avec le résultat de chaque split, 
    # cette fonction fait une moyenne de toutes ces valeurs.
    def moyennes(keys_cv):        
        a1 = grid.cv_results_[keys_cv]
        Avg_key = sum(a1) / float(len(a1))
        #print(Avg_key)
        return Avg_key
    
    # Make predictions
    y_pred = grid.predict(X_test)
    
    # Evaluate model performance
    print()    
    print("++ CV - mean fit time:", round(moyennes('mean_fit_time'),2), 'seg', '++')
    global time
    time = round(moyennes('mean_fit_time'),2)
    print()
    #print("CV - mean_test_accuracy:", round(moyennes('mean_test_accuracy'),3))
    print("Test Accuracy:", round(accuracy_score(y_test, y_pred),3))
    global accuracy_final
    accuracy_final = round(accuracy_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_precision:", round(moyennes('mean_test_precision'),3))
    print("Test precision:", round(precision_score(y_test, y_pred),3))
    global precision_final
    precision_final = round(precision_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_recall:", round(moyennes('mean_test_recall'),3))
    print("Test recall:", round(recall_score(y_test, y_pred),3))
    global recall_final
    recall_final = round(recall_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_f1:", round(moyennes('mean_test_f1'),3))
    print("Test f1:", round(f1_score(y_test, y_pred),3))
    global f1_final
    f1_final = round(f1_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_roc_auc:", round(moyennes('mean_test_roc_auc'),3))
    print("Test roc_auc:", round(roc_auc_score(y_test, y_pred),3))
    global roc_auc_final
    roc_auc_final = round(roc_auc_score(y_test, y_pred),3)
        
    print()
    print("classification_report:")
    print()
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    global cm
    cm=confusion_matrix(y_test,y_pred)
    x= ['0', '1']
    y= ['1', '0']
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in cm]
    global fig_cm
    
    colorscale=[[0.0, 'rgb(97, 132, 232 )'], [.2, 'rgb(97, 232, 228 )'],
            [.4, 'rgb(142, 232, 97 )'], [.6, 'rgb(208, 232, 97 )'],
            [.8, 'rgb(232, 179, 97 )'],[1.0, 'rgb(232, 121, 97)']]
    fig_cm = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale=colorscale)
    layout = {
        "title": "Confusion Matrix", 
        "xaxis": {"title": "Predicted value"}, 
        "yaxis": {"title": "Real value"}
    }
        # add title
    fig_cm.update_layout(title_text='<b>Confusion matrix</b>',xaxis = dict(title='Real value'),
                  yaxis = dict(title='Predicted value')
                     )
    
    
    # adjust margins to make room for yaxis title
    fig_cm.update_layout(margin=dict(t=50, l=50))
    
    try:        
        # Make predictions and  Courbe de ROC
        y_pred = grid.predict(X_test)
        global y_pred_proba
        y_pred_proba =grid.predict_proba(X_test)[:, 1]
        global fpr
        global tpr
        global thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        global fig_roc
        fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'<b>ROC Curve (AUC={auc(fpr, tpr):.4f})</b>', 
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500)
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_roc.update_xaxes(constrain='domain')    

    except: 
        print("Cet estimateur n'a pas la propriété predict_proba pour pouvoir calculer la courbe ROC.")    
    
    
    try:        
        FI = grid.best_estimator_[1].feature_importances_
        
        d_feature = {'Stats':X.columns,
             'FI':FI}
        df_feature = pd.DataFrame(d_feature)

        df_feature = df_feature.sort_values(by='FI', ascending=0)
        print(df_feature)

        fig = px.bar_polar(df_feature, r="FI", theta="Stats",
                           color="Stats", template="plotly_dark",
                           color_discrete_sequence= px.colors.sequential.Plasma_r)
        #fig.show()       
     
    except:
        print()
        print('**********************************************************')
        print("Cet estimateur n'a pas la propriété de feature importances")
        print('**********************************************************')
    
    global df_test_1
    df_test_1 = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'time'])
    df_test_1[modele_donne] = (accuracy_final, precision_final, recall_final, f1_final, roc_auc_final, time)
    
    return accuracy_final, precision_final, recall_final, f1_final, roc_auc_final, time, df_test_1
    
    
metrics_GS = 'roc_auc'


# general_analysis(df)
# univarirate_analysis_A(df)
# correlation_matrix(df)

model = entrainement_evaluation_du_modele(dataml, ExtraTreesClassifier(),{'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'), 'model__min_samples_split':[2, 3, 4]}
   , 'roc_auc')


model_layout = dbc.Card(
            [
                html.H2(precision_final, className="card-title"),
                html.P("Precision ", className="card-text"),
            ],
            body=True,
            color="primary",
            inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},  className="h-100"
        ), dbc.Card(
                [
                    html.H2(accuracy_final, className="card-title"),
                    html.P("Accuracy", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True,style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            ), dbc.Card(
                [
                    html.H2(recall_final, className="card-title"),
                    html.P("Recall", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            ),  dbc.Card(
                [
                    html.H2(f1_final, className="card-title"),
                    html.P("F1 Score", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            ), dbc.Card(
               [
                    html.H2(roc_auc_final, className="card-title"),
                    html.P("ROC AUC", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            ),  dbc.Card(
            [
                html.H2(time, className="card-title"),
                html.P("Temps d'exécution moyen CV (s)", className="card-text"),
            ],
            body=True,
            color="primary",
            inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
        )
        
          

layout =  dcc.Tabs([
        
   dcc.Tab( label='H/V', children=[ 
       
        html.Div([
            html.H2("General"),
            html.H4("Description générale des données"),
       
        html.Div(className='row', children=[
           dbc.Row([dbc.Col(card) for card in model_layout])
           ]),
        
        html.Div(className='row', children=[

        dbc.Row([
            
            dbc.Col( 
                dcc.Graph(
                    figure = fig_roc
            ), style={'display': 'inline-block', 'width':'50%'}),
        
          
            dbc.Col( 
                dcc.Graph(
                    figure = fig_cm
            ), style={'display': 'inline-block',"padding": "2rem 4rem"}) ]) ]),
        
        
   
     ], )  
            
            
         
            ])  ]) 


 

