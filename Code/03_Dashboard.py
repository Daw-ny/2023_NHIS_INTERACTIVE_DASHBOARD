# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
#import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics  import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# load models
#with open('./Data/model_xgb.pickle', 'rb') as f: 
#    xgb = pickle.load(f)

#with open('./Data/model_lgbm.pickle', 'rb') as f: 
#    lgbm = pickle.load(f)

MODELS = {'XGBoost': XGBClassifier,
          'LightGBM': LGBMClassifier
          }

# Incorporate data
df = pd.read_csv('./Data/adult_use_data.csv')

# drop unnecessary columns
data = df.drop(['HHX', 'FMX', 'FPX', 'height.1', 'weight.1', 'bmi.1'], axis = 1)
cri = [data['dm'] == 2,
       data['dm'] == 1]
con = ["DM", "PreDM"]
y_data = np.select(cri, con, default = "Normal")


def get_ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
 
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    
    k2_f = k2[['Variable', 'SHAP_abs', 'Corr']]
    k2_f['SHAP_abs'] = k2_f['SHAP_abs'] * np.sign(k2_f['Corr'])
    k2_f.drop(columns='Corr', inplace=True)
    k2_f.rename(columns={'SHAP_abs': 'SHAP'}, inplace=True)
    
    return k2_f

# Dashboard
app = Dash(external_stylesheets=[dbc.themes.LUX])

# App layout
app.layout = dbc.Container([
    html.Br(),
    dbc.Row([
        html.H1(children='Predictions of Diabetes with NHIS 2018 Data')
    ]),

    dbc.Row([
        html.Hr()
    ]),

    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.H5(children='Histogram of variables')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                html.Label('Variables')
            ]),
            dbc.Row([
                dcc.Dropdown(data.columns,
                    value = 'SEX',
                    id = 'single_variables')
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='histogram-graph')
            ])
        ], width = 6),

        dbc.Col([
            dbc.Row([
                html.H5(children='Scatter plot of 2 continuous variables')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                html.Label('Variables1')
            ]),
            dbc.Row([
                dcc.Dropdown(['AGE_P', 'height', 'weight', 'bmi', 'sleep_hour'],
                            value = 'AGE_P',
                            id = 'scatter_variables1')
            ]),
            dbc.Row([
                html.Label('Variables2')
            ]),
            dbc.Row([
                dcc.Dropdown(['AGE_P', 'height', 'weight', 'bmi', 'sleep_hour'],
                            value = 'height',
                            id = 'scatter_variables2')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                dcc.Graph(figure={}, id='scatter-graph')
            ])
        ], width = 6)
    ]),

    dbc.Row([
        html.H5("Analysis of the ML model's results using ROC and PR curves")
    ]),

    dbc.Row([
        html.P("Select model:")
    ]),

    dbc.Row([
        dcc.Dropdown(
        id='select_model',
        options=list(MODELS.keys()),
        value='XGBoost',
        clearable=False
        )
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dcc.Graph(id="roc_curve_graph")
           ])
        ], width = 6),
        dbc.Col([
            dbc.Row([
                dcc.Graph(id="shap_feature_importance")
           ])
        ], width = 6)
    ]),
    
])

# Add controls to build the interaction
@callback(
    Output(component_id='histogram-graph', component_property='figure'),
    Input(component_id='single_variables', component_property='value')
)
def histogram_graph(col_chosen):
    # Figure  생성
    fig = px.histogram(data,
                        x=col_chosen, 
                        color=y_data,
                        barmode = 'stack') 

    fig.update_layout(
                    template='simple_white',
                    xaxis=dict(title=f'{col_chosen}'),
                    yaxis=dict(title='Count'))

    fig.update_traces(#marker_color= 히스토그램 색, 
                    #marker_line_width=히스토그램 테두리 두깨,                            
                    #marker_line_color=히스토그램 테두리 색,
                    marker_opacity = 0.4,
                    )

    return fig


@callback(
    Output(component_id='scatter-graph', component_property='figure'),
    Input(component_id='scatter_variables1', component_property='value'),
    Input(component_id='scatter_variables2', component_property='value')
)
def scatter_graph(var1, var2):

    fig = px.scatter(data_frame=df,
                x=var1,
                y=var2,
                color=y_data)

    fig.update_layout(
                    template='simple_white',
                    xaxis=dict(title=f'{var1}'),
                    yaxis=dict(title=f'{var2}'))

    fig.update_traces(#marker_color= 히스토그램 색, 
                    #marker_line_width=히스토그램 테두리 두깨,                            
                    #marker_line_color=히스토그램 테두리 색,
                    marker_opacity = 0.4,
                    )

    return fig


@callback(
    Output("roc_curve_graph", "figure"), 
    Output("shap_feature_importance", "figure"), 
    Input('select_model', "value"))
def train_and_display(model_name):
    # 모델의 robust를 확보하기 위해 층화추출로 분리
    X = data.drop("dm",axis=1)
    y = data["dm"]
    X_train, X_test, y_train, y_test =train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state= 42)

    model = MODELS[model_name](random_state = 42)
    model.fit(X_train, y_train)
    
    y_scores = model.predict_proba(X)

    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(y, columns=model.classes_)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()

    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # shap
    shap_values = shap.TreeExplainer(model).shap_values(X_test)

    foo_all = pd.DataFrame()
    for k,v in list(enumerate(model.classes_)):

        foo = get_ABS_SHAP(shap_values[k], X_test)
        foo['class'] = v
        foo_all = pd.concat([foo_all,foo])

    fig_shap = px.bar(foo_all,
                      x='SHAP',
                      y='Variable',
                      color='class')
    
    #범례 삭제하기
    fig_shap.update_layout(showlegend=False)

    return fig, fig_shap


if __name__ == '__main__':
    app.run(debug=True)