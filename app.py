# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:02:13 2019

@author: bberg
"""

# coding: utf-8

import dash
from dash.dependencies import Input, Output, State
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
#import plotly.graph_objs as go
#import plotly.offline as offline 

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import dash_table_experiments as dt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer

#from layout_case import app
#from layout_case import  get_header, get_logo, get_menu, claim_data, claims_input, policy_data, policy_input, app, trends_numbers, make_dash_table, gebruiker, feature_choice,X,y, path
#from Lin_reg_page import test2
#from Introduction_page import overview
#from MLW_page import ML_workflow
#from Data_analysis_page import data_analysis
#from Feature_selection_page import f_selection
#from Training_eval_page import tr_evaluation


### new packages

import os
import base64
from sklearn.linear_model import LogisticRegression


#### layout_case ####

gebruiker = os.path.expanduser('~').split("\\")
gebruiker = gebruiker[-1]
#gebruiker = "bberg"
#__location__ = os.path.realpath(
#    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#path = os.path.dirname(os.path.abspath(__file__))
app = dash.Dash(__name__)
#app = dash.Dash()

#app = dash.Dash()

#STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath('~')), 'static')

#@app.server.route('/static/<resource>')
#def serve_static(resource):
#    return flask.send_from_directory(STATIC_PATH, resource)
  
server = app.server

app.config['suppress_callback_exceptions']=True
#app.config.suppress_callback_exceptions = True


#app.css.config.serve_locally = True
#app.scripts.config.serve_locally = True

#app.config.include_asset_files = True 

# Describe the layout, or the UI, of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),

   # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
])

# # read data
# #dataset = pd.read_csv("dataset.csv")
# claim_final = pd.read_csv("claim_final.csv")
# y_true = claim_final['FraudFound_P']
# del claim_final['FraudFound_P']

#pd.read_csv(path + "\data Case study\claim_data.csv")

claim_data = pd.read_csv("claim_data.csv", delimiter =',')
policy_data = pd.read_csv("policy_dataset_fraud.csv", delimiter =';')
data_load = pd.read_csv("datasetv2.csv", delimiter=';')
y=data_load['FraudFound_P']
del data_load['FraudFound_P']
del data_load['PolicyNumber']
X=pd.get_dummies(data_load)

feature_choice = []
for x in X.columns:
    dict1 = {}
    dict1["label"] = x
    dict1["value"] = x
    feature_choice.append(dict1)

# X_validation = pd.read_csv("X_validation.csv")
#policy_data = pd.read_csv("policy_data.csv")

trends_numbers = pd.read_excel("trend_excel_fraud.xlsx")

claims_input=[]
for x in claim_data.columns:
    dict1 = {}
    dict1["label"] = x
    dict1["value"] = x
    claims_input.append(dict1)

policy_input=[]
for x in policy_data.columns:
    dict2 = {}
    dict2["label"] = x
    dict2["value"] = x
    policy_input.append(dict2)


# fraud_dataset

# reusable componenets
def make_dash_table(df):
    ''' Return a dash definitio of an HTML table for a Pandas dataframe '''
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


# includes page/full view
def get_logo():
    image_filename = "AAA_logo.png"
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    
    
    logo = html.Div([

        html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
#            html.Img(src='https://image.ibb.co/j5vUD8/AAA.jpg',         
              style={
                             'height': '75',
                             'width': '170'
                         })  # 529 × 234 = 2,26
        ], className="two columns padded"),

    ], className="row gs-header")
    return logo

# includes page/full view
# def get_logo():
#     logo = html.Div([
#             html.Img(src='https://image.ibb.co/j5vUD8/AAA.jpg', style={
#                              'height': '110',
#                              'width': '250',
#                              'float': 'right',
#                              'position': 'absolute',
#                          })  # 529 × 234 = 2,26
#         ])
#     return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H4(
                'Triple A - Fraud Detection Case Study')
        ], className="ten columns padded")

    ], className="row gs-header gs-text-header padded")
    return header


def get_menu():

    menu = html.Div([

        dcc.Link('Introduction   ', href='/introduction', className="tab first"),

        dcc.Link('Machine Learning Workflow  ', href='/mlworkflow', className="tab"),

        dcc.Link('Data Analysis   ', href='/thedataset', className="tab"),

        dcc.Link('Feature Selection   ', href='/feature_selection', className="tab"),

        dcc.Link('Training and Evaluation   ', href='/train_evaluate', className="tab")
        
#        dcc.Link('Logistic regression', href='/logistic_reg', className="tab")

    ], className="row ")

    return menu

#### Training_eval_page ####


image_filename = "randomforest.png"
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename2 = "cross_validation.png"
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
                
                       
tr_evaluation = html.Div([  # page 5

        html.Div([

           html.Div([

            # Header

               # Header
               html.Div([
                   html.Div([
                       get_header(),
                   ], className="ten columns"),

                   html.Div([
                       get_logo()
                   ], className="two columns"),

               ], className="row "),

               get_menu(),

            ], className="row "),

            # Row 1

            html.Div([

                html.Div([
                    html.H5(["Introduction Model Building"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            dcc.Markdown('The answer to the question: _"What machine learning algorithm should I use?"_ is always: _"It depends.”_ It depends on the __size__, __quality__, and __nature__ of the data and on the type of prediction problem. Fraud detection is a __supervised__ learning problem, which means that all records are labelled - in this case as either Fraud or No Fraud.' ),
            dcc.Markdown('In this case study we\'ll be using __a Random Forest__ Algorithm since they are relatively easy to implement and understand, while being very powerful.'),
            dcc.Markdown('Below we have set up a machine learning pipeline for you to train and evaluate different Random Forest models on the dataset.'),
            html.Br([]),
            html.Div([

                html.Div([
                    html.H5(["The Random Forest Classifier"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Div([

                html.Div([
                    html.Br([]),
                    dcc.Markdown('Random forests is an __ensemble learning__ method for classification, regression and other tasks, that operate by constructing a _multitude of decision trees_ at training time and outputting the class that is the __mode__ of the classes of the individual trees.'),
                    dcc.Markdown('Random Forests correct for decision trees’ habit of __overfitting__ to their training set by building multiple decision trees with a __subset__ of the features.'),
                    html.Br([]),
                    dcc.Markdown('A Random Forest has about ten parameters to tune. Today we\'ll be using the following:'),
                    dcc.Markdown('1. __Number features Split__: This parameter represents the number of features to consider when looking for the best split.'),
                    dcc.Markdown('2. __Max Depth__: This parameters controls how deep your trees are. Most of the time, it is recommended to limit the depth of the trees if you are dealing with noisy data. The deeper the tree, the more splits it has and it captures more information about the data. '),
                    dcc.Markdown('3. __Max Leaf Nodes__: Select the maximum amount of end nodes in the random forest. Choosing less means that less splits can be made.'),

                ], className="six columns"),

                html.Div([
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    #html.Img(src='https://image.ibb.co/ktcZq8/randomforest.png',
                             style={
                                 'height': '350',
                                 'width': '650',
                                 'top' : '200'
                             }
                             ),
                ], className="six columns"),

            ], className="row"),

            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Evaluate your Model"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([

                    dcc.Markdown(
                        'After you developed a machine learning model for your predictive modelling problem, how do you know if the performance of the model is any good? \
                        We\'ll use historical data to build the model, but in general, you want to develop a model that performs well on __new claims__ that come in, for example, next week. Therefore, you should build a __robust__ model, that __generalizes well__.'),
                    dcc.Markdown('If you evaluate a model on the same data that you used to train and optimize the parameters on, then your model has __already "seen"__ that data through its parameters and the performance will be good on this dataset. However, this model has likely been overfit and will therefore __perform poorly on unseen data.__ '),
                    dcc.Markdown('A popular technique that can be used to evaluate your model and test if it generalizes well is __Cross-Validation__. It is a popular method because it is simple to understand and because it generally results in a __less biased__ or __less (unfair) optimistic__ estimate of the model\'s performance than other methods, such as a simple train/test split. __We will use Cross-Validation to evaluate the performance of our model.__'),
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),
            html.Div([

                html.Div([
                    html.H5(["Using Cross-Validation to evaluate your model and to prevent overfitting"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            html.Div([
                html.Div([

                    dcc.Markdown('Overfitting a model on training data causes __poor model performance__.'),
                    dcc.Markdown('Overfitting happens when a model learns the __detail and noise__ in the training data to the extent that it negatively impacts the performance of the model on new data. The problem is that the noise and details in the training data do not apply to new data and _negatively impact the models ability to generalize_. Generally, __complex__ models are likely to overfit and __too simple models__ don\'t have enough information to make predictions. Therefore, you have to __find a balance__ between a complex and simple model to find one that generalizes well to unseen data.'),
                    html.Br([]),

                    dcc.Markdown('Cross-Validation consists of the following steps:'),
                    dcc.Markdown('1. Shuffle the dataset randomly.'),
                    dcc.Markdown('2. Split the dataset into k groups'),
                    html.Br([]),
                    dcc.Markdown('For each unique group:'),
                    dcc.Markdown('3. Take one group as a hold out or test data set'),
                    dcc.Markdown('4. Take the remaining groups as a training data set'),
                    dcc.Markdown('5. Fit a model on the training set and predict the instances in the test set'),
                    html.Br([]),
                    dcc.Markdown('__In the end you have a prediction for every instance in your data, by a model trained on other groups of data, and you can evaluate how your model performs on unseen data.__'),
                    html.Br([]),
                    dcc.Markdown('Important: Each observation in the data sample is assigned to an __individual group__ and __stays in that group__ for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times. Every time you re-train your model, it will follow the steps above.'),

                ], className="six columns"),

                html.Div([
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),
                    #html.Img(src='https://image.ibb.co/gea13T/cross_validation.png',
                             style={
                                 'height': '300',
                                 'width': '600'
                             }
                             ),
                ], className="six columns"),

            ], className="row"),

            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Build your Random Forest"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Div([
                html.Br([]),

                    dcc.Markdown(
                        'You can use the __dropdown__ and __sliders__ below to change the parameters of your Random Forest model. You can also choose the features that you want to train the model with. When you select Number of features chosen = 10, \
                        then the 10 most important features (based on the algorithmic feature selection from the previous page) will be included in the model. After setting all the parameters you run the model by pushing the __Run Model__ button. When your browser tabs notes __Updating...__ then the model is running. When your browser tab notes __Dash__ your model has finished training and the evaluation tables are updated.'),
                ], className="twelve columns"),

            html.Br([]),
            html.Br([]),


            html.Br([]),
            html.Br([]),

            html.Div([

                html.Div([

                    html.H5([''], id='text5'),

                ], className="four columns", style={'margin-top': 25,'margin-left': 15}),

                html.Div([

                    dcc.Slider(
                        id='slicer_k_best2',
                        min=2,
                        max=126,
                        step=1,
                        value=2,
                    )
                ], className="six columns",style={'margin-top': 25,'margin-left': 15}),

            ], className="row "),

            html.Br([]),
            html.Br([]),

            html.Div([

                html.Div([

                    dcc.Dropdown(
                        id='chosen_features_model',
                        options=feature_choice,
                        multi=True,
                        value=['Age', 'Year']
                    )
                ], className="eleven columns", style={'margin-left': 15}),

            ], className='row'),

            html.Br([]),

            html.Div([
                html.Div([

                 html.H6('', id='esti_text'),

                ], className="four columns", style={'margin-left': 15}),

                html.Div([
                    html.H6('Max Depth', id='dp2'),
                ], className="three columns", style={'margin-left': 40}),
                html.Div([
                    html.H6('', id='depth_text'),
                ], className="four columns", style={'margin-left': 40}),
        ], className = 'row'),

            html.Br([]),

            html.Div([

                    html.Div([

                        dcc.Slider(
                            id='n_features_split',
                            min=1,
                            max=25,
                            step=1,
                            value=10,
                        )

                    ], className="four columns", style={'margin-left': 15}),

                    html.Div([

                        dcc.Dropdown(
                            id='max_depth_slider',
                            options=[
                                {'label': '5', 'value': '5'},
                                {'label': '10', 'value': '10'},
                                {'label': '25', 'value': '25'},
                                {'label': '50', 'value': '50'},
                                {'label': '75', 'value': '75'},
                                {'label': '100', 'value': '100'},
                                {'label': '150', 'value': '150'},
                                {'label': '200', 'value': '200'},
                                {'label': '250', 'value': '250'},
                            ],
                            value='10')

                ] , className="three columns"),

                html.Div([

                    dcc.Slider(
                        id='max_leaf_nodes',
                        min=2,
                        max=150,
                        step=1,
                        value=10,
                    )
                ], className="four columns"),
            ], className="row "),

            html.Br([]),
            html.Div([html.Button('Run Model',id='submit_reg')], className="four columns", style={'margin-left': 15,'margin-bottom': 15}),
            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Model Performance Costs vs. Gains"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),


    
            html.Div([
                html.Br([]),
                dcc.Markdown(
                    'A manager of a fraud department at a large insurance company asks you to calibrate his fraud detection model.'),
                html.Br([]),
                dcc.Markdown(
                    'As mentioned in the introduction, the costs of Fraud are twofold:'),
                dcc.Markdown( '1. The costs of fraudulent claims'),
                dcc.Markdown('2. The fixed and variable costs of running a fraud department'),
                html.Br([]),
                dcc.Markdown(
                    'The manager provides you with the following operational information:'),
                dcc.Markdown('1. The constant costs of running the fraud department = __50.000__'),
                dcc.Markdown('2. The average variable costs per fraud case that is researched by an external party = __750__'),
                dcc.Markdown('3. The gain of finding a fraudulent case = __5000__'),

                html.Br([]),
                dcc.Markdown('The goal is to build a model that maximizes gain for the Fraud Department.'),
                dcc.Markdown('The manager wants you to deliver N cases that have the highest probability of being fraudulent. With the slider below you can select the N records that your model predicted to have the highest probability of fraud. It\'s up to you to find the number of cases that maximizes the result in table 4.'),

                html.Br([]),
                dcc.Markdown(
                    'Below we provided four tables which you can use to test your model\'s performance. The predictions are made by your cross-validated random forest model.'),
                html.Br([]),
                html.Br([]),
                html.Div([

                    html.Div([
                        html.H5([""], id='n_predict_text'),
                    ], className="six columns", style={'margin-left': 15}),

                ], className="row "),
                html.Br([]),
                html.Br([]),

                html.Div([

                    html.Div([

                        dcc.Slider(
                            id='predict_n',
                            min=1,
                            max=10000,
                            step=1,
                            value=10,
                        ),
                        html.Br([]),

                    ], className="five columns", style={'margin-left': 15}),

                ], className="row "),

            ], className="twelve columns"),

            html.Div([

                html.Div([
                    html.H5(["Table 1. N cases with highest probability of fraud"],
                            className="gs-header gs-table-header padded")

                ], className="six columns"),

                html.Div([
                    html.H5(["Table 2. Confusion Matrix"],
                            className="gs-header gs-table-header padded")
                ], className="six columns"),

            ], className="row "),

            html.Br([]),
            html.Div([

                html.Div([
                    dcc.Markdown("The table below shows the N records that your model predicted to have the highest probability of fraud. The left column shows for every record the probability of no fraud, the right column shows the probability of fraud.")

                ], className="six columns"),

                html.Div([
                    dcc.Markdown("The table below shows the classification results in a matrix. It shows the prediction result of all cases predicted by our models compared to the actual outcome. (Fraud – No Fraud)"),
                    dcc.Markdown('The following concepts can help you to evaluate your model\'s performance:'),
                    html.Br([]),
                    dcc.Markdown('__Recall__ = percentage of total fraud cases that your model correctly predicts to be fraud'),
                    dcc.Markdown('__Precision__ = percentage of predicted fraud cases that are actual fraud'),
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Br([]),

            html.Div([

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({'Predicted No Fraud': [0.7700404858299595, 0.8097165991902834, 0.7909238249594813,
                                                 0.7664233576642335, 0.7907542579075426, 0.7850770478507705,
                                                 0.7656123276561233, 0.7956204379562044, 0.7793998377939984,
                                                 0.7769667477696675],
                                           'Predicted Fraud': [0.7700404858299595, 0.8097165991902834, 0.7909238249594813,
                                                 0.7664233576642335, 0.7907542579075426, 0.7850770478507705,
                                                 0.7656123276561233, 0.7956204379562044, 0.7793998377939984,
                                                 0.7769667477696675]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=['Predicted No Fraud', 'Predicted Fraud'],
                        id='DataTable_predict'),

                ], className="six columns"),

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({' ': ['Predicted: No Fraud', 'Predicted: Fraud'],
                                           'Actual: No Fraud': [1000, 400],
                                           'Actual: Fraud': [200, 100]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=[" ", 'Predicted: No Fraud', 'Predicted: Fraud'],

                        id='DataTable_confusion'),

                ], className="six columns"),

            ], className="row "),

            html.Br([]),
            html.Br([]),
            html.Div([

                html.Div([
                    html.H5(["Table 3. Predicted vs. Actual for N selected cases "],
                            className="gs-header gs-table-header padded")

                ], className="six columns"),

                html.Div([
                    html.H5(["Table 4. Costs - Gain - Result"],
                            className="gs-header gs-table-header padded")
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([
                    dcc.Markdown(
                        "The table below compares the actual outcomes with the predicted outcome for the __N cases selected__.")
                ], className="six columns"),

                html.Div([
                    dcc.Markdown(
                        "The table below shows the total __Cost__, __Gain__ and __Result__ for your model."),
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({' ': ['Predicted: Fraud'],
                                           'Actual: No Fraud': [400],
                                           'Actual: Fraud': [100]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=[" ", 'Predicted: Fraud'],

                        id='DataTable_n_predict'),

                ], className="six columns"),

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({'Total Cost': [10000],'Total Gain': [5000], 'Result': [5000]}).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=['Total Cost', 'Total Gain','Result'],
                        id='DataTable_money'),

                ], className="six columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")


@app.callback(
    dash.dependencies.Output('chosen_features_model', 'value'),
    [dash.dependencies.Input('slicer_k_best2', 'value')])
def slicer_k_best2(value):
    #return best
    col = ['Fault_Policy Holder', 'Age', 'Fault_Third Party', 'BasePolicy_Liability', 'VehicleCategory_Sedan', 'RepNumber', 'WeekOfMonth', 'VehicleCategory_Sport', 'WeekOfMonthClaimed', 'DriverRating', 'Year', 'BasePolicy_All Perils', 'BasePolicy_Collision', 'PastNumberOfClaims_none', 'PastNumberOfClaims_2 to 4', 'NumberOfSuppliments_none', 'Deductible', 'VehiclePrice_more than 69000', 'DayOfWeekClaimed_Thursday', 'AgeOfVehicle_6 years', 'DayOfWeekClaimed_Monday', 'DayOfWeekClaimed_Tuesday', 'DayOfWeek_Monday', 'AgeOfVehicle_7 years', 'AddressChange_Claim_no change', 'Make_Pontiac', 'DayOfWeek_Saturday', 'Make_Toyota', 'VehiclePrice_20000 to 29000', 'DayOfWeek_Sunday', 'DayOfWeekClaimed_Wednesday', 'PastNumberOfClaims_1', 'Make_Honda', 'DayOfWeekClaimed_Friday', 'NumberOfSuppliments_1 to 2', 'VehiclePrice_30000 to 39000', 'NumberOfSuppliments_more than 5', 'DayOfWeek_Wednesday', 'AgeOfVehicle_more than 7', 'MaritalStatus_Married', 'DayOfWeek_Tuesday', 'NumberOfSuppliments_3 to 5', 'MaritalStatus_Single', 'Make_Chevrolet', 'Make_Mazda', 'DayOfWeek_Thursday', 'AccidentArea_Rural', 'Sex_Female', 'AddressChange_Claim_2 to 3 years', 'Month_Apr', 'AccidentArea_Urban', 'DayOfWeek_Friday', 'Month_Feb', 'Month_Mar', 'Month_Jan', 'MonthClaimed_Dec', 'Month_Jun', 'MonthClaimed_Nov', 'MonthClaimed_May', 'Month_Dec', 'MonthClaimed_Apr', 'Make_Accura', 'Month_Aug', 'Month_Sep', 'Month_Nov', 'MonthClaimed_Jan', 'MonthClaimed_Mar', 'MonthClaimed_Sep', 'PastNumberOfClaims_more than 4', 'Month_May', 'MonthClaimed_Feb', 'MonthClaimed_Oct', 'Month_Jul', 'Sex_Male', 'Month_Oct', 'MonthClaimed_Jul', 'MonthClaimed_Aug', 'AgeOfVehicle_5 years', 'NumberOfCars_1 vehicle', 'VehicleCategory_Utility', 'VehiclePrice_less than 20000', 'MonthClaimed_Jun', 'PoliceReportFiled_Yes', 'Make_Ford', 'AgeOfVehicle_4 years', 'VehiclePrice_40000 to 59000', 'AgeOfVehicle_new', 'NumberOfCars_2 vehicles', 'Make_VW', 'PoliceReportFiled_No', 'AddressChange_Claim_4 to 8 years', 'NumberOfCars_3 to 4', 'AgentType_Internal', 'DayOfWeekClaimed_Saturday', 'Make_Mercury', 'AgeOfVehicle_3 years', 'MaritalStatus_Widow', 'Days_Policy_Claim_more than 30', 'Make_Saturn', 'AddressChange_Claim_1 year', 'Days_Policy_Accident_more than 30', 'AddressChange_Claim_under 6 months', 'Days_Policy_Claim_15 to 30', 'Make_Dodge', 'Days_Policy_Accident_8 to 15', 'WitnessPresent_Yes', 'Days_Policy_Accident_none', 'AgeOfVehicle_2 years', 'WitnessPresent_No', 'MaritalStatus_Divorced', 'DayOfWeekClaimed_Sunday', 'VehiclePrice_60000 to 69000', 'Make_Nisson', 'Make_Saab', 'Days_Policy_Claim_8 to 15', 'AgentType_External', 'Days_Policy_Accident_15 to 30', 'NumberOfCars_5 to 8', 'Days_Policy_Accident_1 to 7', 'Make_Porche', 'Make_Lexus', 'Make_Jaguar', 'Make_Ferrari', 'Make_BMW', 'Make_Mecedes', 'NumberOfCars_more than 8']
    col = col[:value]
    return col

@app.callback(
    dash.dependencies.Output('text5', 'children'),
    [dash.dependencies.Input('submit_reg','n_clicks'),
     dash.dependencies.Input('chosen_features_model', 'value')])
# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed)
def update_slider2_test(n_clicks,value):
    return 'Number of features chosen = {}'.format(str(len(value)))



@app.callback(
    dash.dependencies.Output('DataTable_n_predict', 'rows'),
    [dash.dependencies.Input('submit_reg','n_clicks'),
     dash.dependencies.Input('predict_n', 'value')
    ],
    state=[dash.dependencies.State('max_leaf_nodes', 'value'),
     dash.dependencies.State('max_depth_slider', 'value'),
     dash.dependencies.State('chosen_features_model', 'value'),
     dash.dependencies.State('n_features_split', 'value')])
def run_predict_datatable_cv(n_clicks,predict_n,max_leaf_nodes,  max_depth_slider, chosen_features_model, n_features_split):

    forest = RandomForestClassifier(n_estimators=25, max_features= min(n_features_split,max(len(chosen_features_model),2)), random_state=1000,class_weight='balanced', max_leaf_nodes=int(max_leaf_nodes), max_depth=int(max_depth_slider))

    #select column based on dropdown selectors
    if len(chosen_features_model) == 1 or len(chosen_features_model) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model]


    proba = cross_val_predict(forest, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba['Actual'] = y

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]

    tp = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] >= 0.5)]

    if tp.empty:
        tp = 0
    else:
        tp = tp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fp = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] >= 0.5)]
    if fp.empty:
        fp = 0
    else:
        fp = fp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]


    tn = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] <= 0.5)]
    if tn.empty:
        tn = 0
    else:
        tn = tn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fn = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] <= 0.5)]

    if fn.empty:
        fn = 0
    else:
      fn = fn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    conf_mat = pd.DataFrame(data=[[fp],[tp]],columns=['Predicted: Fraud'])
    conf_mat[" "]=['Actual: No Fraud', 'Actual: Fraud']

    return conf_mat.to_dict('records')

@app.callback(
    dash.dependencies.Output('DataTable_money', 'rows'),
    [dash.dependencies.Input('submit_reg','n_clicks'),
     dash.dependencies.Input('predict_n', 'value')
    ],
    state = [dash.dependencies.State('max_leaf_nodes', 'value'),
     dash.dependencies.State('max_depth_slider', 'value'),
     dash.dependencies.State('chosen_features_model', 'value'),
     dash.dependencies.State('n_features_split', 'value')])
def run_predict_datatable_cv2(n_clicks,predict_n, max_leaf_nodes, max_depth_slider, chosen_features_model, n_features_split):

    forest = RandomForestClassifier(n_estimators=25, random_state=1000, max_features=min(n_features_split,max(len(chosen_features_model),2)), class_weight='balanced', max_leaf_nodes=int(max_leaf_nodes), max_depth=int(max_depth_slider))

    #select column based on dropdown selectors
    if len(chosen_features_model) == 1 or len(chosen_features_model) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model]

    proba = cross_val_predict(forest, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba['Actual'] = y

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]


    tp = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] >= 0.5)]

    if tp.empty:
        tp = 0
    else:
        tp = tp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fp = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] >= 0.5)]
    if fp.empty:
        fp = 0
    else:
        fp = fp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    tn = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] <= 0.5)]
    if tn.empty:
        tn = 0
    else:
        tn = tn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fn = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] <= 0.5)]

    if fn.empty:
        fn = 0
    else:
      fn = fn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]


    total_cost = (tp+fp+tn+fn) * 800 + 50000
    total_profit = (tp * 5000)
    result = total_profit-total_cost

    conf_mat = pd.DataFrame({ 'Total Cost':[total_cost], 'Total Gain' : [total_profit], 'Result' : [result]})

    return conf_mat.to_dict('records')

@app.callback(
    dash.dependencies.Output('n_predict_text', 'children'),
    [dash.dependencies.Input('predict_n', 'value')])
def update_slider_predict(value):
    return 'Selected {} cases for the fraud department'.format(str(value))

@app.callback(
    dash.dependencies.Output('DataTable_predict', 'rows'),
    [dash.dependencies.Input('submit_reg','n_clicks'),
     dash.dependencies.Input('predict_n', 'value')
    ],
    state = [dash.dependencies.State('max_leaf_nodes', 'value'),
     dash.dependencies.State('max_depth_slider', 'value'),
     dash.dependencies.State('chosen_features_model', 'value'),
     dash.dependencies.State('n_features_split', 'value')])
def run_model_predict(n_clicks,predict_n, max_leaf_nodes, max_depth_slider, chosen_features_model, n_features_split):

    forest = RandomForestClassifier(n_estimators=25, max_features=min(n_features_split,max(len(chosen_features_model),2)),random_state=1000, class_weight='balanced', max_leaf_nodes=int(max_leaf_nodes), max_depth=int(max_depth_slider))

    #select column based on dropdown selectors
    if len(chosen_features_model) == 1 or len(chosen_features_model) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model]

    # Fit forest
    #forest.fit(X_train, y)

    proba = cross_val_predict(forest, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]


    return proba.to_dict('records')

@app.callback(
    dash.dependencies.Output('DataTable_confusion', 'rows'),
    [dash.dependencies.Input('submit_reg','n_clicks')],
    state=[dash.dependencies.State('max_leaf_nodes', 'value'),
     dash.dependencies.State('max_depth_slider', 'value'),
     dash.dependencies.State('chosen_features_model', 'value'),
     dash.dependencies.State('n_features_split', 'value')])
def confusion_matrixv2(n_clicks,max_leaf_nodes, max_depth_slider, chosen_features_model, n_features_split):

    #select column based on dropdown selectors
    if len(chosen_features_model) == 1 or len(chosen_features_model) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model]

    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=25, max_features=min(n_features_split,max(len(chosen_features_model),2)), random_state=1000, class_weight='balanced', max_leaf_nodes=int(max_leaf_nodes), max_depth=int(max_depth_slider))


    # Fit forest
    forest.fit(X_train, y)

    #train model and output datatable

    y_pred = cross_val_predict(forest, X_train, y, cv=StratifiedKFold(10, random_state=1000))

    conf_mat = confusion_matrix(y, y_pred)

    conf_mat = pd.DataFrame(data=conf_mat,columns=['Predicted: No Fraud', 'Predicted: Fraud'])
    conf_mat[" "]=['Actual: No Fraud', 'Actual: Fraud']


    return conf_mat.to_dict('records')

@app.callback(
    dash.dependencies.Output('depth_text', 'children'),
    [dash.dependencies.Input('max_leaf_nodes', 'value')])
def update_slider1_test(value):
    return 'Max Leaf Nodes = {} Splits'.format(str(value))

@app.callback(
    dash.dependencies.Output('dp2', 'children'),
    [dash.dependencies.Input('max_depth_slider', 'value')])
def update_slider1_test2(value):
    return 'Max Depth = {}'.format(str(value))

@app.callback(
    dash.dependencies.Output('esti_text', 'children'),
    [dash.dependencies.Input('n_features_split', 'value')])
def update_slider1_test3(value):
    return 'N - Features Split = {} Features'.format(str(value))




#### Feature_selection_page ####


f_selection = html.Div([  # page 4

        html.Div([

            # Header
            html.Div([
                html.Div([
                    get_header(),
                ], className="ten columns"),

                html.Div([
                    get_logo()
                ], className="two columns"),

            ], className="row "),

            get_menu(),
            # Row 1

            html.Div([

                html.Div([
                    html.H5(["Algorithmic Feature Selection"],
                            className="row gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            html.Div([
                dcc.Markdown('_"Which features should you use to create a predictive model?"_'),
                dcc.Markdown('This is a difficult question that may require deep knowledge of the problem domain. It is possible to automatically select those features in your data that are most useful or most relevant for the problem you are working on. This is a process called: _algorithmic feature selection_.'),
                dcc.Markdown('Feature selection methods can be used to identify and remove __unneeded__, __irrelevant__ and __redundant__ attributes from data that do not contribute to the __accuracy__ of a predictive model or may in fact decrease the accuracy of the model. _Fewer attributes is desirable_ because it reduces the _complexity_ of the model. Complex models are more likley to overfit and harder to explain.'),
                dcc.Markdown(
                        'A  Random Forest Algorithm can be used to rank features based on how much information they contain. It can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure.'),
                html.Br([]),
            ], className="tekstblok"),


            html.Div([

                html.Div([
                    html.H5(["Features ranked on Importance "],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            html.Br([]),

            html.Div([
                dcc.Markdown(
                    'Use the slider below to select more or less features or choose features from the dropdown:'),
            ], className="tekstblok"),
                        
            html.Div([

                html.Div([

                  html.P([''], id='text4'),

                ], className="four columns"),

            ], className="row"),

            html.Br([]),
            html.Br([]),

            html.Div([

                html.Div([

                    dcc.Slider(
                        id='slicer_k_best',
                        min=2,
                        max=126,
                        step=1,
                        value=5,
                    )
                ], className="eight columns"),
            ], className="row"),

            html.Br([]),

            html.Div([
                    dcc.Markdown('__Click on the dropdown to select more features from the list:__'),
            ], className="tekstblok"),

            html.Br([]),

            html.Div([

                html.Div([
                    dcc.Dropdown(
                        id='chosen_features',
                        options=feature_choice,
                        multi=True,
                        value=['Age', 'Year']
                    )
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            dcc.Graph(
                id='graph_feature_selection',
                style={'height': '65vh'})
            #],
            # className="twelve columns"),

        ], className="subpage")

    ], className="page")

@app.callback(
    dash.dependencies.Output('text4', 'children'),
    [dash.dependencies.Input('slicer_k_best', 'value')])
def update_slider2_test(value):
    return 'Number of features chosen = {}'.format(str(value))

@app.callback(
    dash.dependencies.Output('chosen_features', 'value'),
    [dash.dependencies.Input('slicer_k_best', 'value')])
def slicer_k_best2(value):
    #return best
    col = ['Fault_Policy Holder', 'Age', 'Fault_Third Party', 'BasePolicy_Liability', 'VehicleCategory_Sedan', 'RepNumber', 'WeekOfMonth', 'VehicleCategory_Sport', 'WeekOfMonthClaimed', 'DriverRating', 'Year', 'BasePolicy_All Perils', 'BasePolicy_Collision', 'PastNumberOfClaims_none', 'PastNumberOfClaims_2 to 4', 'NumberOfSuppliments_none', 'Deductible', 'VehiclePrice_more than 69000', 'DayOfWeekClaimed_Thursday', 'AgeOfVehicle_6 years', 'DayOfWeekClaimed_Monday', 'DayOfWeekClaimed_Tuesday', 'DayOfWeek_Monday', 'AgeOfVehicle_7 years', 'AddressChange_Claim_no change', 'Make_Pontiac', 'DayOfWeek_Saturday', 'Make_Toyota', 'VehiclePrice_20000 to 29000', 'DayOfWeek_Sunday', 'DayOfWeekClaimed_Wednesday', 'PastNumberOfClaims_1', 'Make_Honda', 'DayOfWeekClaimed_Friday', 'NumberOfSuppliments_1 to 2', 'VehiclePrice_30000 to 39000', 'NumberOfSuppliments_more than 5', 'DayOfWeek_Wednesday', 'AgeOfVehicle_more than 7', 'MaritalStatus_Married', 'DayOfWeek_Tuesday', 'NumberOfSuppliments_3 to 5', 'MaritalStatus_Single', 'Make_Chevrolet', 'Make_Mazda', 'DayOfWeek_Thursday', 'AccidentArea_Rural', 'Sex_Female', 'AddressChange_Claim_2 to 3 years', 'Month_Apr', 'AccidentArea_Urban', 'DayOfWeek_Friday', 'Month_Feb', 'Month_Mar', 'Month_Jan', 'MonthClaimed_Dec', 'Month_Jun', 'MonthClaimed_Nov', 'MonthClaimed_May', 'Month_Dec', 'MonthClaimed_Apr', 'Make_Accura', 'Month_Aug', 'Month_Sep', 'Month_Nov', 'MonthClaimed_Jan', 'MonthClaimed_Mar', 'MonthClaimed_Sep', 'PastNumberOfClaims_more than 4', 'Month_May', 'MonthClaimed_Feb', 'MonthClaimed_Oct', 'Month_Jul', 'Sex_Male', 'Month_Oct', 'MonthClaimed_Jul', 'MonthClaimed_Aug', 'AgeOfVehicle_5 years', 'NumberOfCars_1 vehicle', 'VehicleCategory_Utility', 'VehiclePrice_less than 20000', 'MonthClaimed_Jun', 'PoliceReportFiled_Yes', 'Make_Ford', 'AgeOfVehicle_4 years', 'VehiclePrice_40000 to 59000', 'AgeOfVehicle_new', 'NumberOfCars_2 vehicles', 'Make_VW', 'PoliceReportFiled_No', 'AddressChange_Claim_4 to 8 years', 'NumberOfCars_3 to 4', 'AgentType_Internal', 'DayOfWeekClaimed_Saturday', 'Make_Mercury', 'AgeOfVehicle_3 years', 'MaritalStatus_Widow', 'Days_Policy_Claim_more than 30', 'Make_Saturn', 'AddressChange_Claim_1 year', 'Days_Policy_Accident_more than 30', 'AddressChange_Claim_under 6 months', 'Days_Policy_Claim_15 to 30', 'Make_Dodge', 'Days_Policy_Accident_8 to 15', 'WitnessPresent_Yes', 'Days_Policy_Accident_none', 'AgeOfVehicle_2 years', 'WitnessPresent_No', 'MaritalStatus_Divorced', 'DayOfWeekClaimed_Sunday', 'VehiclePrice_60000 to 69000', 'Make_Nisson', 'Make_Saab', 'Days_Policy_Claim_8 to 15', 'AgentType_External', 'Days_Policy_Accident_15 to 30', 'NumberOfCars_5 to 8', 'Days_Policy_Accident_1 to 7', 'Make_Porche', 'Make_Lexus', 'Make_Jaguar', 'Make_Ferrari', 'Make_BMW', 'Make_Mecedes', 'NumberOfCars_more than 8']
    col = col[:value]
    return col



@app.callback(
    dash.dependencies.Output('graph_feature_selection', 'figure'),
    [dash.dependencies.Input('chosen_features', 'value')])
def feature_selection(features):

    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=50, random_state=1000, class_weight= 'balanced')

    #select column based on dropdown selectors
    if len(features) == 1 or len(features) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[features]

    # Fit forest

    forest.fit(X_train, y)

    # calculate feature importances

    importances = forest.feature_importances_

    # calculate std

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    indices = np.argsort(importances)[::-1]

    list_col = []
    for f in range(X_train.shape[1]):
        list_col.append(list(X_train.columns)[indices[f]])

    list4 = []
    std_list = []
    for x in indices:
        list4.append(importances[x])
        std_list.append(std[x])

    # Plot the feature importances of the forest
    df = pd.DataFrame({"importance": list4, 'columns': list_col, 'std':std_list})

    return {
            'data': [
                {'x': df['columns'], 'y': df["importance"], 'type': 'bar',
                 'error_y' : {'type': 'data', 'array' : df['std'], 'visible' : False}
                 },
            ],
            'layout': {'margin': {'l': 40, 'r': 40, 't': 30, 'b': 150}, 'title': 'Relative Feature Importance',
                       'yaxis' : {'title':'Feature Importance'}}
        }



#### Data_analysis_page ####

data_analysis = html.Div([ # page 3


        html.Div([

            # Header
            html.Div([
                html.Div([
                    get_header(),
                ], className="ten columns"),

                html.Div([
                    get_logo()
                ], className="two columns"),

            ], className="row "),

            get_menu(),

            # Row 1

            html.Div([

                html.Div([
                    html.H5(["Data Analysis"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),
            html.Br([]),
            dcc.Markdown("The insurer's administration system consists of two datasets. The first dataset (__Policy Data Set__) contains information about the policy and contains 14 columns and 12.335 records. Some variables are for example: The brand of the car, the policy holder's gender and age, the vehicle price, etc. This data is all gathered by the insurance company when you __subscribe to their policy.__ "),
            html.Br([]),
            dcc.Markdown("The second dataset (__Claims Data Set__) contains information about the claim and contains 18 columns and 12.335 entries. This data is gathered by the insurance company __at the moment a policy holder files a claim__. Some of the variables are for example: The date, the kind of area the accident took place, if a police report was filed, if witnesses were present, etc."),

            html.Br([]),
            dcc.Markdown(
                "Select a dataset in the dropdown menu below to show the data in the table below. The dropdown menu below the datatable shows all column for the chosen dataset. The __left__ graph shows the __distribution__ of the chosen column in the dataset. The __right__ graph shows the __fraud percentage for every category__ in that column."),
            html.Br([]),
            dcc.Markdown("This step in the process is called __Exploratory Data Analysis__ (EDA). Exploratory Data Analysis (EDA) refers to using techniques to display data in such a way that interesting features will become apparent. Unlike classical methods which usually begin with an assumed model for the data, EDA techniques are used to encourage the data to __suggest models__ that might be appropriate and to __find features__ with potentially high predictive power."),
            html.Br([]),
            dcc.Markdown("_Try to find some features which have potentially high predictive power:_"),

            html.Br([]),

            dcc.Markdown("__Select Data Set:__"),

            dcc.Dropdown(
                id='dropdown_dataset',
                options=[
                    {'label': 'Claims Data Set', 'value': 'Claims Data Set'},
                    {'label': 'Policy Data Set', 'value': 'Policy Data Set'}],
                value='Claims Data Set'
            ),

            html.Br([]),
            html.Br([]),

            html.Div([
                html.Div([
                    dt.DataTable(
                        rows=claim_data.to_dict('records'),
                        sortable=True,
                        editable=False,
                        filterable=False,
                        id='DataTable'),
                ], className="twelve columns"),
            ], className="row "),

            html.Br([]),
            html.Br([]),

            dcc.Markdown("__Select Column to Visualize:__"),

            html.Br([]),

            dcc.Dropdown(
                id='dropdown_column_viz',
                options=[claims_input
                ],
                value=claims_input[0]
            ),

            html.Br([]),

            html.Div([

                html.Div([
                    dcc.Graph(
                        id='graph_data_viz1',
                        style={'height': '55vh'})
                ], className="six columns"),

                html.Div([
                    dcc.Graph(
                        id='graph_data_viz2',
                        style={'height': '55vh'})
                ], className="six columns"),


            ], className="row "),


            html.Br([]),

        ], className="subpage")

    ], className="page")

## Call back graph_data_viz1

@app.callback(
    dash.dependencies.Output('graph_data_viz1', 'figure'),
    [dash.dependencies.Input('dropdown_column_viz', 'value'),
     dash.dependencies.Input('dropdown_dataset', 'value')])
def update_figure_company(column, dataset):

    #Bar chart met gemiddelde van sector en gemiddelde van branche met naam van de branche

    # list with categorical

    if type(column) == dict:
        column = column['value']


    if dataset == 'Policy Data Set':
        data = policy_data
    else:
        data = claim_data

    data['count'] = 1

    grouper = data.groupby(column).count().reset_index()

    categories = list(grouper[column])
    values = list(grouper['count'])

    return {
            'data': [
                {'x': categories, 'y': values, 'type': 'bar'},
            ],
            'layout': {'margin': {'l': 40, 'r': 40, 't': 30, 'b': 150}, 'title': 'Category Distribution',
                       'yaxis' : {'title':'Count'}}
        }

# Callback graph_data_viz
@app.callback(
    dash.dependencies.Output('graph_data_viz2', 'figure'),
    [dash.dependencies.Input('dropdown_column_viz', 'value'),
     dash.dependencies.Input('dropdown_dataset', 'value')])
def update_figure_company(column, dataset):

    #Bar chart met gemiddelde van sector en gemiddelde van branche met naam van de branche

    # list with categorical

    if type(column) == dict:
        column = column['value']


    if dataset == 'Policy Data Set':
        data = policy_data
    else:
        data = claim_data

    data['count'] = 1

    grouper = data.groupby(column).sum().reset_index()

    if column == 'FraudFound_P':
        categories = ['0','1']
        values = [0, 100]
    else:
        categories = grouper['FraudFound_P'] / grouper['count'] * 100
        values = list(categories)
        categories = list(grouper[column])

    return {
            'data': [
                {'x': categories, 'y': values, 'type': 'bar'},
            ],
            'layout': {'margin': {'l': 40, 'r': 40, 't': 30, 'b': 150}, 'title': 'Distribution of Fraud Cases per Category',
                       'yaxis' : {'title':'Percentage of total claims marked as Fraudulent %'}}
        }

# Callback DataTable
@app.callback(
    dash.dependencies.Output('DataTable', 'rows'),
    [dash.dependencies.Input('dropdown_dataset', 'value')])
def set_rows_dt(dataset):

    if dataset == 'Policy Data Set':
        return policy_data.to_dict('records')
    else:
        return claim_data.to_dict('records')

#update columns dataset tab

@app.callback(
    dash.dependencies.Output('dropdown_column_viz', 'options'),
    [dash.dependencies.Input('dropdown_dataset', 'value')])
def set_columns(dataset):

    if dataset == 'Policy Data Set':
        g = policy_input
       # del g[0]['FraudFound_P']
        return g
    else:
        k = claims_input
     #   del k[0]['FraudFound_P']
        return k

@app.callback(
    dash.dependencies.Output('dropdown_column_viz', 'value'),
    [dash.dependencies.Input('dropdown_dataset', 'value')])
def set_columns(dataset):

    if dataset == 'Policy Data Set':
        return policy_input[2]
    else:
        return claims_input[0]



#### MLW_page ####


image_filename = "Flowchart.png"
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
ML_workflow = html.Div([  # page 2

    html.Div([

        # Header
        html.Div([
            html.Div([
                get_header(),
            ], className="ten columns"),

            html.Div([
                get_logo()
            ], className="two columns"),

        ], className="row "),

        get_menu(),

        html.Div([

            html.Div([
                html.H5(["The Machine Learning Workflow"],
                        className="gs-header gs-table-header padded")
            ], className="twelve columns"),

        ], className="row "),
        html.Br([]),
        html.Br([]),

        dcc.Markdown("The machine learning workflow typically follows these five steps:"),
        html.Br([]),
        dcc.Markdown("1. __Data Preparation__ Encompasses data cleaning and transformation in order to be able to work with the data. Often data has to be collected from multiple datasources. In this step the data is put into the right shape, format and quality."),
        html.Br([]),
        dcc.Markdown("2. __Exploratory Data Analysis__ (EDA) refers to using techniques to display data in such a way that interesting features will become apparent. Unlike classical methods which usually begin with an assumed model for the data,  \
                     EDA techniques are used to encourage the data to suggest models that might be appropriate. The reason for the heavy reliance on graphics is that by its very nature the main role of EDA is to open-mindedly explore. We are usually looking for structures and unexpected insights."),
        html.Br([]),
        dcc.Markdown("3. __Feature selection__ refers to the selection of features that will be used in the model. The quality and quantity of the features in your model will largely influence the predictive power of your model."),
        html.Br([]),
        dcc.Markdown("4. __Modelling / Training__ refers to choosing algorithms and building your predictive models."),
        html.Br([]),
        dcc.Markdown("5. __Evaluation__ refers to testing how well your models perform. Usually many different models are tried and evaluated (optimization feedback loop)."),
        html.Br([]),
    
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
#        html.Img(src='https://image.ibb.co/mhjNaJ/Screen_Shot_2018_06_14_at_11_53_11.png',
                             style={
                                 'height': '325',   #1990 × 439
                                 'width': '1000',
                                 'left' : '150'
                             }),
        ], className="subpage")

    ], className="page")



#### Introduction_page ####

overview = html.Div([  # page 1


        html.Div([

            # Header
            html.Div([
                html.Div([
                    get_header(),
                ], className="ten columns"),

                html.Div([
                  get_logo()
            ], className="two columns"),

            ], className="row "),

            get_menu(),

            # Row 3

            html.Div([

                html.Div([

                    html.H5('1. Introduction',
                            className="gs-header gs-text-header padded"),

                    html.Br([]),
                    dcc.Markdown("In this case study we focus on a major issue for insurance companies: __Fraudulent Claims__. \
                    Fraud, or criminal deception, is a costly problem for insurance companies and causes large losses."),
                    
                    dcc.Markdown("The cost of fraud is twofold:"),
                    dcc.Markdown("_1. The direct cost of covering expenses for fraudulent claims._"),
                    dcc.Markdown("_2. The cost of fraud prevention and detection._"),

                    dcc.Markdown("Furthermore, these fraudulent claims also have a social-economic impact since the insurance companies costs for fraud are passed on to the policy holders by means of a higher premium."),

                    dcc.Markdown("Insurance companies want to keep their premiums as low as possible compared to competitors to \
        attract new customers and increase marketshare. Therefore, fraud detection is important for insurance companies."),

                ], className="six columns"),

                html.Div([
                    html.H5(["2. About Fraud Detection"],
                            className="gs-header gs-table-header padded"),
                    html.Br([]),
                    dcc.Markdown("Intelligent fraud detection systems that use data have been around for while. \
                    However, substantial improvements have been made in recents years with the introduction of new and improved Machine Learning Algorithms."),
                    
                    dcc.Markdown("Furthermore, insurers are becoming data-driven organizations which collect large amounts of data. \
                    This data can be collected from their own systems, but can also be retrieved from open data sources or bought from data providers."),
                    
                    dcc.Markdown("The current challenge for insurers is to leverage the recent __advancements in machine learning__ and \
                    the __ever-growing data available__ to improve their fraud detection systems."),
                ], className="six columns"),

            ], className="row"),

            # Row 4

            html.Div([

                html.Div([
                    html.H5('3. About the Case',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),

                dcc.Markdown("In this case study we'll be working with a __publicly available dataset__ found on Oracle's website. \
                This dataset is often used in scientific papers in the fraud prediction domain. \
                The dataset contains policy and claims data from an car insurance company that contains both fraudulent and valid claims. \
                The data is from an American car insurance company and dates from around the year 2000."),

                dcc.Markdown("In this case study you are going to develop a __prototype predictive machine learning model__ that can be embedded in a car insurer's fraud detection system."),

                html.Br([]),
                dcc.Markdown(" __In this case you will:__"),
                dcc.Markdown("1. Be divided in teams of 3 - 4. "),
                dcc.Markdown("2. Do some analysis / visualization to become familiar with the datasets. "),
                dcc.Markdown("3. Develop a model that, as accurately as possible, identifies fraudulent claims."),
                dcc.Markdown("4. Optimize your detection model based on cost / gain per fraud case."),
                html.Br([]),

                ], className="six columns"),


                html.Div([
                    html.H5("4. Dutch Insurance Fraud in Numbers",
                            className="gs-header gs-table-header padded"),
                ], className="six columns"),

                html.Br([]),

                html.Br([]),

                html.Br([]),

                html.Br([]),

                html.Div([
                    dt.DataTable(
                        rows=trends_numbers.to_dict('records'),
                        id='DataTable_fissa',
                        sortable=True,
                        editable=False,
                        filterable=False,
                        columns=['Jaar', '2016']),
                ], className="six columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")


#### Lin_reg_page ####

image_filename = "LogReg_regularization.png"
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
image_filename2 = "cross_validation.png"
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
      
test2 = html.Div([  # page 5

        html.Div([

           html.Div([

            # Header

               # Header
               html.Div([
                   html.Div([
                       get_header(),
                   ], className="ten columns"),

                   html.Div([
                       get_logo()
                   ], className="two columns"),

               ], className="row "),

               get_menu(),

            ], className="row "),

            # Row 1

            html.Div([

                html.Div([
                    html.H5(["Introduction Model Building"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            dcc.Markdown('The answer to the question: _"What machine learning algorithm should I use?"_ is always: _"It depends.”_ It depends on the __size__, __quality__, and __nature__ of the data and on the type of prediction problem. Fraud detection is a __supervised__ learning problem, which means that all records are labelled - in this case as either Fraud or No Fraud.' ),
            dcc.Markdown('In this case study we\'ll be using __Logistic Regression__ ....'),
            dcc.Markdown('Below we have set up a pipeline for you to calculate and evaluate different Logistic Regression models on the dataset.'),
            html.Br([]),
            html.Div([

                html.Div([
                    html.H5(["Logistic Regression"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Div([

                html.Div([
                    html.Br([]),

                    dcc.Markdown('In Logistic Regression .....'),
                    html.Br([]),
                    dcc.Markdown('For the Logistic Regression model you will set two parameters:'),
                    dcc.Markdown('1. __Coefficient for the regression__:  '),
                    dcc.Markdown('2. __Penalty for the regression__: Using regularization in the Logistic Regression model adds a penalty on the different parameters to reduce the freedom of the model. \
                                 Using this regularization will ensure that the model is less likely to fit the noise of the training data and therefore improve the generalization abilities of the model.\
                                 You can choose between L1 and L2 regularization. In L1 regularization the penalty is equal to the sum of the __absolute__ value of the coefficients, \
                                 which will shrink some of the chosen parameters to zero, whereas L2 penalties the sum of the __squared__ value of the coefficients, forcing the parameters to be relatively small'),

                ], className="six columns"),

                html.Div([
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    #html.Img(src='https://image.ibb.co/ktcZq8/randomforest.png',
                             style={
                                 'height': '265',
                                 'width': '650',
                                 'top' : '200'
                             }
                             ),
                ], className="six columns"),

            ], className="row"),

            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Evaluate your Model"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([

                    dcc.Markdown(
                        'After you developed a machine learning model for your predictive modelling problem, how do you know if the performance of the model is any good? \
                         We\'ll use historical data to build the build, but in general, you want to develop a model that performs well on __new claims__ that come in, for example, next week. Therefore, you should build a __robus__ model, that __generalizes well__.'),
                    dcc.Markdown('If you evaluate a model on the same data that you used to train and optimze the parameters on, then your model has __already "seen"__ that data through its parameters and the performance will be good on this dataset. However, this model has likely been overfit and will therefore __perform poorly on unseen data.__ '),
                    dcc.Markdown('A popular technique that can be used to evaluate your model and test if it generalizes well is __Cross-Validation__. It is a popular method because it is simple to understand and because it generally results in a __less biased__ or __less (unfair) optimistic__ estimate of the model\'s performance than other methods, such as a simple train/test split.'),
                ], className="twelve columns"),

            ], className="row "),
                            
            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Using Cross-Validation to evaluate your model and to prevent overfitting"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Br([]),


            html.Div([
                html.Div([

                    dcc.Markdown('Overfitting a model on training data causes __poor model performance__.'),
                    dcc.Markdown('Overfitting happens when a model learns the __detail and noise__ in the training data to the extent that it negatively impacts the performance of the model on new data. The problem is that the noise and details in the training data do not apply to new data and _negatively impact the models ability to generalize_. Generally, __complex__ models are likely to overfit and __too simple models__ don\'t have enough information to make predictions. Therfore, you have to __find a balance__ between a complex and simple model to find a that generalizes well to unseen data.'),
                    html.Br([]),

                    dcc.Markdown('Cross-Validation follows the following steps:'),
                    dcc.Markdown('1. Shuffle the dataset randomly.'),
                    dcc.Markdown('2. Split the dataset into k groups'),
                    html.Br([]),
                    dcc.Markdown('For each unique group:'),
                    dcc.Markdown('3. Take one group as a hold out or test data set'),
                    dcc.Markdown('4. Take the remaining groups as a training data set'),
                    dcc.Markdown('5. Fit a model on the training set and predict the instances in the test set'),
                    html.Br([]),
                    dcc.Markdown('__In the end you have predicted all the instances in your data.__'),
                    html.Br([]),
                    dcc.Markdown('Important: Each observation in the data sample is assigned to an __individual group__ and __stays in that group__ for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times. Every time you re-train your model, it will follow the steps above.'),

                ], className="six columns"),

                html.Div([
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),
                    #html.Img(src='https://image.ibb.co/gea13T/cross_validation.png',
                             style={
                                 'height': '300',
                                 'width': '600'
                             }
                             ),
                ], className="six columns"),

            ], className="row"),

            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Build your Logistic Regression model"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            html.Div([
                html.Br([]),

                    dcc.Markdown(
                        'You can use the __dropdown__ menus below to change the parameters of your Logistic Regression model. In the first dropdown menu you can select the parameters you want to use for the Logistic Regression. \
                        With the other two you can set the desired coefficient and penalty for the regression. After setting these parameters you run the model by pushing the __submit regression__ button. When your browser tabs notes __Updating...__ then the model is running. \
                        When your browser tab notes __Dash__ your model calculation has finished and the evaluation tables are updated.'),
                ], className="twelve columns"),

            html.Br([]),

            html.Div([

                html.Div([

                    html.H5(['Nothing'], id='text52'),

                ], className="four columns", style={'margin-top': 25,'margin-left': 15}),

            ], className="row "),

            html.Br([]),
            html.Br([]),

            html.Div([

                html.Div([

                    dcc.Dropdown(
                        id='chosen_features_model_2',
                        options=feature_choice,
                        multi=True,
                        value=['Age', 'Year']
                    )
                ], className="ten columns", style={'margin-left': 15,'margin-right': 15})
                
            ], className='row'),
            html.Br([]),
            html.Div([html.Div([
                    html.H5(
                        'Coefficient for the regression:'),
                ], className="four columns", style={'margin-left': 15}),
            html.Div([
                    html.H5(
                        'Penalty for the regression:'),
                ], className="four columns", style={'margin-left': 30})], className='row'),
            html.Br([]),
            html.Div([                    
                html.Div([

                    dcc.Dropdown(
                        id='Coeff',
                        options=[{'label': '1', 'value' :"1"},{'label': '2', 'value' :"2"},{'label': '3', 'value' :"3"}],
                        multi=False,
                        value="1"
                    )
                ], className="four columns", style={'margin-left': 15,'margin-right': 15}),
                    html.Div([

                    dcc.Dropdown(
                        id='penalty',
                        options=[{'label':'l1','value':'l1'},{'label':'l2','value':'l2'}],
                        multi=False,
                        value='l1'
                    )
                ], className="four columns", style={'margin-left': 15,'margin-right': 15})], className='row'),
                    # Button to submit the regression
                    html.Br([]),
                html.Div([html.Div([html.Button('Submit regression',id='submit')], className="four columns", style={'margin-left': 15})], className="row"),



            html.Br([]),

            html.Div([
                html.Div([

                 html.H6(''),

                ], className="four columns"),
                html.Div([
                    html.H6(''),
                ], className="four columns"),
        ], className = 'row'),

            html.Br([]),

            html.Br([]),

            html.Div([

                html.Div([
                    html.H5(["Model Performance Costs vs. Gains"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),


    
            html.Div([
                html.Br([]),
                dcc.Markdown('A manager of a fraud department at a large insurance company asks you to calibrate his fraud detection model.'),
                html.Br([]),
                dcc.Markdown('As mentioned in the introduction, the costs of Fraud are twofold:'),
                dcc.Markdown('1. The costs of fraudulent claims'),
                dcc.Markdown( '2. The fixed and variable costs of running a fraud department'),
                
                html.Br([]),
                dcc.Markdown('The manager provides you with the following operational information:'),
                dcc.Markdown('1. The constant costs of running the fraud department = __50.000__'),
                dcc.Markdown('2. The average variable costs per fraud case that is researched by an external party = __750__'),
                dcc.Markdown('3. The gain of finding a fraudulent case = __5000__'),

                html.Br([]),
                dcc.Markdown('The goal is to build a model that maximizes gain for the Fraud Department.'),
                dcc.Markdown('The manager wants you to deliver N cases that have the highest chance on being fraudulent. With the slider below you can select the N records that your model predicted to have the highest chance on Fraud. It\'s up to you to find the number of cases that maximizes the result in table 4.'),

                html.Br([]),
                dcc.Markdown(
                    'Below we provided four tables which you can use to test your model\'s performance:'),
                html.Br([]),
                html.Br([]),
                html.Div([

                    html.Div([
                        html.H5([""], id='n_predict_text2'),
                    ], className="six columns", style={'margin-left': 15}),

                ], className="row "),
                html.Br([]),
                html.Br([]),

                html.Div([

                    html.Div([

                        dcc.Slider(
                            id='predict_n_2',
                            min=1,
                            max=10000,
                            step=1,
                            value=10,
                        ),
                        html.Br([]),
                        html.Br([]),

                    ], className="five columns", style={'margin-left': 15}),

                ], className="row "),

            ], className="twelve columns"),

            html.Div([

                html.Div([
                    html.H5(["Table 1. N cases with highest chance on Fraud"],
                            className="gs-header gs-table-header padded")

                ], className="six columns"),

                html.Div([
                    html.H5(["Table 2. Confusion Matrix"],
                            className="gs-header gs-table-header padded")
                ], className="six columns"),

            ], className="row "),

            html.Br([]),
            html.Div([

                html.Div([
                    dcc.Markdown("The table below shows the N records that your model predicted to have the highest chance on Fraud. The left columns shows for every records the change for No Fraud, the right column shows the chance for Fraud.")

                ], className="six columns"),

                html.Div([
                    dcc.Markdown("The table below shows the classification results in a matrix. It shows the prediction result of all cases predicted by our models compared to the actual outcome. (Fraud – No Fraud)"),
                    dcc.Markdown('The following concepts can help you to evaluate your model\'s performance:'),
                    html.Br([]),
                    dcc.Markdown('__Recall__ = percentage of total fraud cases that your model correctly predicts to be fraud'),
                    dcc.Markdown('__Precision__ = percentage of predicted fraud cases that are actual fraud'),
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Br([]),

            html.Div([

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({'Predicted No Fraud': [0.7700404858299595, 0.8097165991902834, 0.7909238249594813,
                                                 0.7664233576642335, 0.7907542579075426, 0.7850770478507705,
                                                 0.7656123276561233, 0.7956204379562044, 0.7793998377939984,
                                                 0.7769667477696675],
                                           'Predicted Fraud': [0.7700404858299595, 0.8097165991902834, 0.7909238249594813,
                                                 0.7664233576642335, 0.7907542579075426, 0.7850770478507705,
                                                 0.7656123276561233, 0.7956204379562044, 0.7793998377939984,
                                                 0.7769667477696675]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=['Predicted No Fraud', 'Predicted Fraud'],
                        id='DataTable_predict2'),

                ], className="six columns"),

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({' ': ['Predicted: No Fraud', 'Predicted: Fraud'],
                                           'Actual: No Fraud': [1000, 400],
                                           'Actual: Fraud': [200, 100]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=[" ", 'Predicted: No Fraud', 'Predicted: Fraud'],

                        id='DataTable_confusion2'),

                ], className="six columns"),

            ], className="row "),

            html.Br([]),
            html.Br([]),
            html.Div([

                html.Div([
                    html.H5(["Table 3. Predicted vs. Actual for N selected cases "],
                            className="gs-header gs-table-header padded")

                ], className="six columns"),

                html.Div([
                    html.H5(["Table 4. Costs - Gain - Result"],
                            className="gs-header gs-table-header padded")
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([
                    dcc.Markdown(
                        "The table below compares the actual outcomes with the predicted for the __N cases selected__.")
                ], className="six columns"),

                html.Div([
                    dcc.Markdown(
                        "The table below shows the total __Cost__, __Gain__ and __Result__ for your model."),
                ], className="six columns"),

            ], className="row "),

            html.Br([]),

            html.Div([

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({' ': ['Predicted: Fraud'],
                                           'Actual: No Fraud': [400],
                                           'Actual: Fraud': [100]
                                           }).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=[" ", 'Predicted: Fraud'],

                        id='DataTable_n_predict2'),

                ], className="six columns"),

                html.Div([
                    dt.DataTable(
                        rows=pd.DataFrame({'Total Cost': [10000],'Total Gain': [5000], 'Result': [5000]}).to_dict('records'),
                        sortable=False,
                        editable=False,
                        filterable=False,
                        columns=['Total Cost', 'Total Gain','Result'],
                        id='DataTable_money2'),

                ], className="six columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")

@app.callback(
    dash.dependencies.Output('text52', 'children'),
    [dash.dependencies.Input('submit','n_clicks')],
# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed)
    state=[dash.dependencies.State('chosen_features_model_2', 'value')])
def update_slider2_test(n_clicks,value):
    return 'Number of features chosen = {}'.format(str(len(value)))

# No Fraud, Fraud count table for the Fraud cases ivestigated
@app.callback(
    dash.dependencies.Output('DataTable_n_predict2', 'rows'),
    [dash.dependencies.Input('submit','n_clicks'),
     dash.dependencies.Input('predict_n_2', 'value')
    ],
# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed or slider # of cases checked is changed)
    state=[dash.dependencies.State('Coeff', 'value'),
     dash.dependencies.State('penalty', 'value'),
     dash.dependencies.State('chosen_features_model_2', 'value')])
def run_predict_datatable_cv(n_clicks,predict_n,Coeff,penalty,chosen_features_model_2):
    
    clf_LR = LogisticRegression(C=int(Coeff),penalty=penalty,class_weight='balanced')
    #select column based on dropdown selectors
    if len(chosen_features_model_2) == 1 or len(chosen_features_model_2) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model_2]


    proba = cross_val_predict(clf_LR, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba['Actual'] = y

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]

    tp = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] >= 0.5)]

    if tp.empty:
        tp = 0
    else:
        tp = tp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fp = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] >= 0.5)]
    if fp.empty:
        fp = 0
    else:
        fp = fp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]


    tn = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] <= 0.5)]
    if tn.empty:
        tn = 0
    else:
        tn = tn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fn = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] <= 0.5)]

    if fn.empty:
        fn = 0
    else:
      fn = fn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    conf_mat = pd.DataFrame(data=[[fp],[tp]],columns=['Predicted: Fraud'])
    conf_mat[" "]=['Actual: No Fraud', 'Actual: Fraud']

    return conf_mat.to_dict('records')


# Profit table
@app.callback(
    dash.dependencies.Output('DataTable_money2', 'rows'),
    [dash.dependencies.Input('submit','n_clicks'),
     dash.dependencies.Input('predict_n_2', 'value')
    ],

# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed or slider # of cases checked is changed)    
    state=[dash.dependencies.State('Coeff', 'value'),
     dash.dependencies.State('penalty', 'value'),
     dash.dependencies.State('chosen_features_model_2', 'value')])
def run_predict_datatable_cv2(n_clicks,predict_n,Coeff,penalty,chosen_features_model_2):

    clf_LR = LogisticRegression(C=int(Coeff),penalty=penalty,class_weight='balanced')
  
    #select column based on dropdown selectors
    if len(chosen_features_model_2) == 1 or len(chosen_features_model_2) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model_2]

    proba = cross_val_predict(clf_LR, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba['Actual'] = y

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]


    tp = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] >= 0.5)]

    if tp.empty:
        tp = 0
    else:
        tp = tp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fp = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] >= 0.5)]
    if fp.empty:
        fp = 0
    else:
        fp = fp.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    tn = proba[(proba['Actual'] == 0) & (proba['Predicted Fraud'] <= 0.5)]
    if tn.empty:
        tn = 0
    else:
        tn = tn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]

    fn = proba[(proba['Actual'] == 1) & (proba['Predicted Fraud'] <= 0.5)]

    if fn.empty:
        fn = 0
    else:
      fn = fn.groupby('Actual').count().reset_index()['Predicted Fraud'][0]


    total_cost = (tp+fp+tn+fn) * 800 + 50000
    total_profit = (tp * 5000)
    result = total_profit-total_cost

    conf_mat = pd.DataFrame({ 'Total Cost':[total_cost], 'Total Gain' : [total_profit], 'Result' : [result]})

    return conf_mat.to_dict('records')

# Predicted chances for the # of cases investigated (in descending order)
@app.callback(
    dash.dependencies.Output('DataTable_predict2', 'rows'),
    [dash.dependencies.Input('submit','n_clicks'),
     dash.dependencies.Input('predict_n_2', 'value')
    ],
# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed or slider # of cases checked is changed)    
    state=[dash.dependencies.State('Coeff', 'value'),
     dash.dependencies.State('penalty', 'value'),
     dash.dependencies.State('chosen_features_model_2', 'value')])
def run_model_predict(n_clicks,predict_n,Coeff,penalty,chosen_features_model_2):

    clf_LR = LogisticRegression(C=int(Coeff),penalty=penalty,class_weight='balanced')
  
    #select column based on dropdown selectors
    if len(chosen_features_model_2) == 1 or len(chosen_features_model_2) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model_2]

    # Fit forest
    #forest.fit(X_train, y)

    proba = cross_val_predict(clf_LR, X_train, y, cv=StratifiedKFold(10, random_state=1000), method='predict_proba')

    proba = pd.DataFrame(data=proba,  # values
            columns = ["Predicted No Fraud", 'Predicted Fraud'])

    proba = proba.sort_values(by='Predicted Fraud', ascending=False)

    proba = proba[0:predict_n]


    return proba.to_dict('records')

# Confusion matrix oll data
@app.callback(
    dash.dependencies.Output('DataTable_confusion2', 'rows'),
        [dash.dependencies.Input('submit','n_clicks')],
# If state variable is changed the output is NOT calculated
# This is only done if the input is changed (Submit Button is pressed or slider # of cases checked is changed)
    state=[dash.dependencies.State('Coeff', 'value'),
     dash.dependencies.State('penalty', 'value'),
     dash.dependencies.State('chosen_features_model_2', 'value')])
def confusion_matrixv2(n_clicks,Coeff,penalty,chosen_features_model_2):

    # Build a forest and compute the feature importances
    clf_LR = LogisticRegression(C=int(Coeff),penalty=penalty,class_weight='balanced')
  
    #select column based on dropdown selectors
    if len(chosen_features_model_2) == 1 or len(chosen_features_model_2) == 0:
        X_train = X[['Age', 'Year']]
    else:
        X_train = X[chosen_features_model_2]

    #train model and output datatable

    y_pred = cross_val_predict(clf_LR, X_train, y, cv=StratifiedKFold(10, random_state=1000))

    conf_mat = confusion_matrix(y, y_pred)

    conf_mat = pd.DataFrame(data=conf_mat,columns=['Predicted: No Fraud', 'Predicted: Fraud'])
    conf_mat[" "]=['Actual: No Fraud', 'Actual: Fraud']


    return conf_mat.to_dict('records')

@app.callback(
    dash.dependencies.Output('n_predict_text2', 'children'),
    [dash.dependencies.Input('predict_n_2', 'value')])
def update_slider_predict(value):
    return 'Selected {} cases for fraud department'.format(str(value))






#### Page layout ####
    
newsReviews = html.Div([  # page 7

        html.Div([

            # Header
        html.Div([
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),


            ], className="row ")

        ], className="subpage")

    ], className="page")

noPage = html.Div([  # 404

    html.P(["404 Page not found"])

    ], className="no-page")



# Update page
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/' or pathname == '/introduction':
        return overview
    elif pathname == '/mlworkflow':
        return ML_workflow
    elif pathname == '/thedataset':
        return data_analysis
    elif pathname == '/feature_selection':
        return f_selection
    elif pathname == '/train_evaluate':
        return tr_evaluation
    elif pathname == '/logistic_reg':
        return test2
    else:
        return noPage
    
external_css = ["https://codepen.io/jdmol/pen/EMZRpE.css",
                "https://codepen.io/jdmol/pen/OqWEaV.css",
                "https://codepen.io/jdmol/pen/rRjKoJ.css",
                "https://codepen.io/jdmol/pen/pYRKGw.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://codepen.io/jdmol/pen/PLWavP.js",
               "https://codepen.io/jdmol/pen/ywgEdz.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


if __name__ == '__main__':
    app.run_server(debug=True, processes =True)
