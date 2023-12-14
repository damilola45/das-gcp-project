#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import pickle
from dash.dependencies import Input, Output
from wordcloud import WordCloud
from main_model import predict_mortality_data, predict_mortality_data_rf, predict_mortality_data_xgb, predict_mortality_data_rid
from data_cleaning import clean_data

df = pd.read_csv("https://storage.googleapis.com/mbcc/datasets/us_chronic_disease_indicators.csv")
#df = pd.read_csv("c:/data/us_chronic_disease_indicators.csv")
# Clean the data using the clean_data function from data_cleaning.py
cleaned_data = clean_data(df)
# df contains your dataset
df_filtered = df[df['topic'].isin(['Cancer', 'Cardiovascular Disease', 'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease', 'Diabetes'])]


df3 = pd.read_csv("https://storage.googleapis.com/mbcc/datasets/us_chronic_disease_indicators.csv")
#df3 = df3[df3.locationdesc.isin(['West Virginia', 'New Hampshire', 'Maine', 'Montana', 'Rhode Island', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska', 'Vermont', 'Wyoming', 'Nebraska'])]
df3 = df3[~df3['locationdesc'].isin(['United States'])]
df3 = df3[(df3.topic.isin(['Alcohol','Cardiovascular Disease', 'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease', 'Diabetes']))]
with open('columns_order.pkl', 'rb') as f:
    columns_order = pickle.load(f)
# Filter data for all mortality-related questions
mortality_questions = df_filtered[df_filtered['question'].str.contains('Mortality')]
# Filter rows based on desired topics
desired_topics = [
    'Cancer', 'Cardiovascular Disease', 'Chronic Kidney Disease',
    'Chronic Obstructive Pulmonary Disease', 'Diabetes'
]
filtered_df = df[df['topic'].isin(desired_topics)]

# Create options for the dropdown based on the filtered DataFrame
dropdown_options = [{'label': topic, 'value': topic} for topic in filtered_df['topic'].unique()]

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Filtered data initialization
filtered_data = df[
    (df['topic'] == 'Chronic Obstructive Pulmonary Disease')
    & (df['question'] == 'Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years')
    & (df['datavaluetypeid'] == 'CRDRATE')
    & (df['locationabbr'] != 'US')
]

filtered_data.loc[:, 'datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')

if not filtered_data.empty:
    fig9 = px.box(
        filtered_data, x='locationabbr', y='datavalue',
        title='Statewise distribution of Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years'
    )
    fig9.update_xaxes(title='States')
    fig9.update_yaxes(title='Crude Rate (%)')
server = app.server

colors = {
    'background': '#1c77ac',
    'text': '#7FDBFF'
}
mdesired_topics = [
    'Cardiovascular Disease', 'Chronic Kidney Disease',
    'Chronic Obstructive Pulmonary Disease', 'Diabetes',
    'Older Adults', 'Overarching Conditions', 'Asthma', 'Alcohol'
]

mortality_questions = df[df['question'].str.contains('Mortality')]
mortality_topics = mortality_questions['topic'].unique()

valid_topics = list(set(mdesired_topics).intersection(mortality_topics))

box_dropdown_options = [{'label': topic, 'value': topic} for topic in valid_topics]
# Dropdown menu options
options = [{'label': question, 'value': question} for question in mortality_questions['question'].unique()]

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='US Harmful Chronic Diseases Dashboard (2011 - 2020)',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    
    html.Div([
        html.H2(children='Chronic Diseases Visualization & Model Prediction Dashboard', style={
            'textAlign': 'center',
            'color': colors['text'],
            'fontSize': '24px'
        })
    ]),
    
    html.Div([
        dcc.Tabs(id='tabs-example', value='tab1', children=[
            dcc.Tab(label='Home', value='tab1', style={'fontWeight': 'bold', 'fontSize': '18px'}),
            dcc.Tab(label='Most 5 Harmful Chronic Diseases', value='tab2', style={'fontWeight': 'bold', 'fontSize': '18px'}),
            dcc.Tab(label='Mortality Trend Analysis', value='tab3', style={'fontWeight': 'bold', 'fontSize': '18px'}),
            dcc.Tab(label='Mortality Race/Ethnicity Analysis', value='tab4', style={'fontWeight': 'bold', 'fontSize': '18px'}),
            dcc.Tab(label='Total Number of Cases Across US States', value='tab5', style={'fontWeight': 'bold', 'fontSize': '18px'}),
            dcc.Tab(label='Model Prediction', value='tab6', style={'fontWeight': 'bold', 'fontSize': '18px'}),
        ],
        style={'backgroundColor': '#f0f0f0'})
    ]),
    
    html.Div(id='tabs-content-example')
])

df_filtered = df[df['topic'].isin(['Cancer', 'Cardiovascular Disease', 'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease', 'Diabetes'])]
# Dropdown for selecting topics
topic_dropdown = dcc.Dropdown(
    options=[{'label': topic, 'value': topic} for topic in df_filtered['topic'].unique()],
    value='Cancer',
    id='topic-dropdown2')
state_dropdown = dcc.Dropdown(
    options=[{'label': state, 'value': state} for state in df_filtered['locationabbr'].unique()],
    value='NY',
    id='state-dropdown'
)

gender_distribution_graph = dcc.Graph(id='gender-distribution-graph')

@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def update_tab_content(tab_name):
    if tab_name == 'tab1':
        graph1 = dcc.Graph(figure=fig)  #the figure you want to display
        # the column containing data sources
        html.Div('this is a mortality trend analysis'),        
        data_source_counts = df['datasource'].value_counts().reset_index()
        data_source_counts.columns = ['DataSource', 'Count']

        # Select the top ten data sources by frequency
        top_ten_data_sources = data_source_counts.head(10)

        # Creating a pie chart to show the distribution of the top ten data sources
        fig5 = px.pie(
        top_ten_data_sources, values='Count', names='DataSource',
        title='Top Ten Data Sources Distribution'
)
        fig5.update_traces(textposition='inside', textinfo='percent+label'),  # Add percentage labels
   
        dcc.Graph(id='pie-chart', figure=fig5)
        
        #'Per Capita Alcohol Consumption' is the topic for alcohol consumption data
        selected_topic = 'Alcohol'

        # Filter the DataFrame for the selected topic excluding 'United States'
        wordfiltered_df = df[(df['topic'] == selected_topic) & (df['locationdesc'] != 'United States')]

        # Generate a dictionary of state names and their respective alcohol consumption values
        state_alcohol_dict = wordfiltered_df.set_index('locationdesc')['datavalue'].to_dict()

        # Create the WordCloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(state_alcohol_dict)

        # Convert word cloud to an array
        wordcloud_array = wordcloud.to_array()

        # Create a Plotly figure
        fig0 = px.imshow(wordcloud_array)
        fig0.update_layout(title='Alcohol Consumption per State')

        dcc.Graph(figure=fig0)
        
        return html.Div([
            html.Div([html.P("Assumption - The assumption is to identify and understand the dominant or most prevalent chronic diseases within the dataset.", style={ 'font-size': '24px'}), graph1, html.P("Findings: The observation from the stacked bar graph highlights the prevalence of specific chronic diseases. This finding indicates that Cardiovascular Disease, Cancer, Chronic Kidney Disease, Chronic Obstructive Pulmonary Disease, and Diabetes are among the top five chronic diseases depicted in the dataset. This implies a substantial occurrence or reporting frequency of these specific diseases compared to others, drawing attention to their prominence within the analyzed data.", style={ 'font-size': '24px','padding': '10px','border': '4px solid white'})]),  # Div for graph1 with a paragraph
            html.Div([html.P("Assumption - All the datasoures have contributed equally to the dataset ", style={ 'font-size': '24px'}), dcc.Graph(id='pie-chart', figure=fig5), html.P("Findings: The Pie-Chart presentation reveals a distinct distribution among data sources, showcasing BRFSS as the predominant contributor, followed closely by NVSS. with BRFSS having a substantial percentage , it indicating its extensive coverage or influence in providing data, while NVSS follows suit, albeit with a slightly lesser but still substantial contribution. This finding illuminates the disproportionate impact or prevalence of BRFSS in sourcing data, suggesting its potential pivotal role in the dataset compared to other sources like NVSS.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'})]),  # Div for pie chart with a paragraph
            html.Div([html.P("Assumption - We assume that the most populated state will have highest Alcohol consumption rate", style={ 'font-size': '24px'}), dcc.Graph(figure=fig0), html.P("Findings: Based on the Word Cloud analysis, Louisiana and Tennessee emerge as the top states with the highest recorded alcohol consumption per capita. This visual representation emphasizes the prevalence of alcohol consumption in these specific states, signifying a noteworthy pattern of higher alcohol use when compared to other states.", style={'font-size': '24px','border': '4px solid white','padding': '10px'})]),  # Div for fig0 with a paragraph
            ])
    elif tab_name == 'tab2':
        return html.Div([
            html.H3('Gender Distribution by Topic and States'),
            html.Div('Assumption - The observation of a higher number of reported cases among females compared to males in densely populated states such as California and New York suggests a potential correlation between gender-based health-seeking behaviors ', style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
            topic_dropdown,            
            state_dropdown,          
            gender_distribution_graph,
            html.Div("In highly populated states such as California and New York, a notable trend has been observed: a higher incidence of reported cases among females compared to males.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),           
        ])
    elif tab_name == 'tab3':
        return html.Div([
            html.H1("Mortality Trend Analysis"),
            html.Div("Assumption - The assumption is that there exists a significant correlation between the passage of time and the increased mortality rates associated with chronic illnesses.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),            
            dcc.Dropdown(
                id='question-dropdown',
                options=options,
                value=options[0]['value']  # Default value
    ),
            dcc.Graph(id='mortality-trend'),
            html.Div("Insight - The analysis of the graph strongly indicates a notable and concerning pattern: a consistent increase in mortality rates associated with chronic diseases over the years. This trend is visibly portrayed through the rising trajectory of mortality rates observed across multiple years in the graph.  This observation raises awareness about the persistent and possibly worsening impact of chronic diseases on mortality rates, underscoring the need for focused interventions and healthcare strategies to address this concerning trend.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
        ])
    elif tab_name == 'tab4':
        return html.Div([
            #html.H1('Topic Distribution by Gender and Race'),
            html.Div("Assumption - To find out if any specific Race/Ethnicity has the highest Mortality rate among all the diseases/topics.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
            #dcc.Dropdown(
                #id='topic-dropdown',
                #options=[{'label': topic, 'value': topic} for topic in df['topic'].unique()],
                #value=df['topic'].unique()[0]
            #),
            #dcc.Graph(id='gender-race-graph')
            dcc.Dropdown(
            id='topic-dropdown3',
            options=[
                {'label': topic, 'value': topic} for topic in mdesired_topics
            ],
            value='Chronic Obstructive Pulmonary Disease',
            style={'width': '50%'}
        ),
        dcc.Graph(id='box-plot'),
        html.Div("Insight - In the overall dataset, we found that Oklahoma, California, and Michigan had diverse Race/Ethnicity in terms of death rate and that also implies the diversity existence in those states as well.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
    ])

#else:
    #app.layout = html.Div("No data to plot.")
        
    elif tab_name == 'tab5':
        return html.Div([
            #html.H3('Content of Tab 5'),
            html.P("Assumption - The assumption is that areas with denser populations may face increased challenges in managing and controlling the spread or reporting of cases due to higher interaction rates.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
            #dcc.Graph(figure=fig8)
            dcc.Dropdown(
            id='topic-dropdown4',
            options=dropdown_options,
            value='Cancer',  # Default value
            multi=False,
            style={'width': '50%'}
    ),
    dcc.Graph(id='choropleth-map'),
    html.P("In our analysis of the Chloropleth map, a striking pattern emerged: states with larger populations, such as Texas and California, exhibited notably higher numbers of recorded cases. This observation highlights a correlation between population density and case counts. The map vividly illustrates that regions with denser populations tend to report a higher incidence of cases, emphasizing the impact of population size on the prevalence of the recorded cases. This insight sheds light on the potential relationship between population density and the spread or reporting of cases, suggesting the need for targeted measures in densely populated areas to manage and mitigate the impact of such cases effectively.", style={ 'font-size': '24px', 'border': '4px solid white','padding': '10px'}),
        ])
    elif tab_name == 'tab6':
        return html.Div([
           html.H1("Chronic Diseases Mortality Rate Prediction(per 1,000,000 persons)", style={ 'font-size': '24px','padding': '5px'}),
           dcc.Markdown("### **Key Data Points:**", style={ 'font-size': '24px','padding': '5px'}),
           html.P(" 1: To make a balanced dataset, datavalueunit is scaled to 1M for all the states that reported at 100,000 and 1,000", style={ 'font-size': '24px', 'border': '4px solid white','padding': '5px'}),
           html.P(" 2: Excluded Cancer Diseases, as the Average Numbers were randomly spread between the years and not in a proportional manner", style={ 'font-size': '24px', 'border': '4px solid white','padding': '5px'}),
           html.P(" 3: Selected 5 Most Harmful Chronic Diseases for building the model ", style={ 'font-size': '24px', 'border': '4px solid white','padding': '5px'}),
           html.P(" 4: Onehotencoding provided better accuracy compared to normal encoding in all the models ", style={ 'font-size': '24px', 'border': '4px solid white','padding': '5px'}),
           html.P(" 5: Random Forest and XGBoost had lower MSE and better R2 value during Prediction ", style={ 'font-size': '24px', 'border': '4px solid white','padding': '5px'}),   
        
    html.Label('Select Location:'),  # Label for location dropdown
    # Inputs for categorical features
    dcc.Dropdown(
        id='locationdesc-dropdown',
        options=[{'label': loc, 'value': loc} for loc in df3['locationdesc'].unique()],
        value=df3['locationdesc'].iloc[0],
        multi=False,
        placeholder="Select Location",
        style={'margin-bottom': '10px'}  # Add margin to the bottom
    ),

    html.Label('Select Topic:'),  # Label for topic dropdown        
    dcc.Dropdown(
        id='topic-dropdown',
        options=[{'label': topic, 'value': topic} for topic in df3['topic'].unique()],
        value=df3['topic'].iloc[0],
        multi=False,
        placeholder="Select Diseases/Topic",
        style={'margin-bottom': '10px'}  # Add margin to the bottom
    ),

    html.Label('Select Year:'),  # Label for ye
    # Input for numeric feature
    dcc.Input(
        id='yearend-input',
        type='number',
        value=df3['yearend'].iloc[0],
        placeholder="Enter Yearend",
        style={'margin-bottom': '10px'}  # Add margin to the bottom
    ),

    html.Br(),
    # Button for making prediction
    html.Button(
    'PREDICT',
    id='predict-button',
    style={
        'margin-top': '10px',  # Adjust top margin
        'margin-bottom': '10px',  # Adjust bottom margin
        'width': '200px',  # Set button width
        'height': '40px',  # Set button height
        'font-size': '16px',  # Set font size
        'color': 'black',  # Set text color
     #   'background-color': '#007BFF',  # Set background color
        'border-radius': '5px',  # Set border radius for rounded corners
    }
),
    #FOr displaying the prediction manually
    #html.P('Linear Regression -> Mortality Rate: 2736 & r2: 0.0180 ... Ridge Regression -> Mortality Rate: 2773 & r2: 0.0180 ... Random Forest -> Mortality Rate: 105 & r2: 0.9956 ... XGBoost -> Mortality Rate: 117 & r2: 0.9960', style={ 'font-size': '24px', 'border': '4px solid black','padding': '10px'}),
    # Output for displaying predictions
    html.Div(id='prediction-output'),
    
        ])
# Callback to update the plot based on dropdown selection
@app.callback(
    Output('mortality-trend', 'figure'),
    [Input('question-dropdown', 'value')]
)
def update_plot(selected_question):
    question_data = mortality_questions[mortality_questions['question'] == selected_question]
    grouped_data = question_data.groupby('yearstart')['datavalue'].sum().reset_index()
    
    fig = px.line(grouped_data, x='yearstart', y='datavalue', title=f'Five-Year Trend: {selected_question}')
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='Total Mortality')
    return fig
@app.callback(
    Output('gender-distribution-graph', 'figure'),
    [Input('topic-dropdown2', 'value'), Input('state-dropdown', 'value')]
)
def update_graph(selected_topic, selected_state):
    filtered_df = df_filtered[(df_filtered['topic'] == selected_topic) & (df_filtered['locationabbr'] == selected_state)]
    filtered_df['gender'] = filtered_df['stratification1'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')
    filtered_df = filtered_df[filtered_df['gender'].isin(['Male', 'Female'])]
    gender_counts = filtered_df.groupby('gender').size().reset_index(name='count')
    fig1 = px.bar(gender_counts, x='gender', y='count', color='gender',
                 labels={'count': 'Count', 'stratification1': 'Gender'},
                 title=f"Gender Distribution for {selected_topic}")
    return fig1
# Extracting relevant columns
state_disease_counts = df.groupby(['locationabbr', 'topic']).size().unstack(fill_value=0)
# Plotting the distribution
fig = go.Figure()

for topic in state_disease_counts.columns:
    fig.add_trace(go.Bar(
        x=state_disease_counts.index,
        y=state_disease_counts[topic],
        name=topic
    ))

fig.update_layout(
    barmode='stack',
    title='Distribution of Topics by State',
    xaxis=dict(title='State'),
    yaxis=dict(title='Count'),
    legend=dict(title='Chronic Disease Topic', x=1, y=1, traceorder='normal')
)    
# Filter for 'Heavy drinking among adults aged >= 18 years' and remove 'Overall' and gender data
#race_ethnicity_heavy_drinking = df[
    #(df['question'] == 'Heavy drinking among adults agged >= 18 years') & 
    #(~df['stratification1'].isin(['Overall', 'Male', 'Female']))
#]

# Create the choropleth map
#fig8 = px.choropleth(
    #race_ethnicity_heavy_drinking,
    #locations='locationabbr',
    #locationmode='USA-states',
    #color='datavalue',
    #hover_data=['stratification1'],  # Race/Ethnicity as hover data
    #scope='usa',
    #labels={'datavalue': 'Percentage of Heavy Drinking Adults', 'locationabbr': 'State'},
    #title='Heavy Drinking Adults and their Race/Ethnicity by State'
#)

# Update layout for better visualization
#fig8.update_layout(
   # geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)'),
   # coloraxis_colorbar=dict(title='Percentage'),
#)
#@app.callback(
   # dash.dependencies.Output('choropleth-map2', 'figure'),
   # [dash.dependencies.Input('topic-dropdown', 'value')]
#)

# Callback to update the choropleth map based on the selected topics
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('topic-dropdown4', 'value'),
     Input('topic-dropdown4', 'value')]
)
def update_choropleth_map(selected_topic1, selected_topic):
    # Use the selected topic from either dropdown, prioritize topic1-dropdown
    selected_topic = selected_topic1 if selected_topic1 else selected_topic

    # Filter the DataFrame based on the selected topic
    filtered_df = df[df['topic'] == selected_topic]

    # Group by state abbreviation and sum the cases
    state_cases = filtered_df.groupby('locationabbr')['datavalue'].sum().reset_index()

    # Create the choropleth map using Plotly Express
    fig = px.choropleth(
        state_cases,
        locations='locationabbr',  # DataFrame column with locations
        color='datavalue',  # DataFrame column with values
        locationmode='USA-states',  # Set to plot as US States
        color_continuous_scale='Viridis',  # Color scale
        scope='usa',  # Scope of the map is USA
        labels={'datavalue': f'Total Cases of {selected_topic}'},  # Label for the color bar
        title=f'Total Number of {selected_topic} Cases Across US States'
    )
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    [Input('topic-dropdown3', 'value')]
)
def update_box_plot(selected_topic):
    # Filter the DataFrame based on the selected topic
    filtered_data = df[
        (df['topic'] == selected_topic)
        #& (df['question'].str.contains('Mortality'))
        & (df['datavaluetypeid'] == 'CRDRATE')
        & (df['stratificationcategory1'] == 'Race/Ethnicity')
        & (df['locationabbr'] != 'US')
    ]

    filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')

    if not filtered_data.empty:
        fig = px.box(
            filtered_data, x='locationabbr', y='datavalue',
            title=f'Statewise distribution of Mortality with {selected_topic} by Race/Ethnicity'
        )
        fig.update_xaxes(title='States')
        fig.update_yaxes(title='Crude Rate (%)')
        return fig
    else:
        return {}
    
import os
linear_r2_value = float(os.environ.get('linear_r2', '0.000'))
linear_rid_value = float(os.environ.get('linear_rid', '0.000'))
linear_rf_value = float(os.environ.get('linear_rf', '0.000'))
linear_xgb_value = float(os.environ.get('linear_xgb', '0.000'))

# Define callback to update prediction output
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('locationdesc-dropdown', 'value'),
    dash.dependencies.State('topic-dropdown', 'value'),
    dash.dependencies.State('yearend-input', 'value')]
)
def update_prediction(n_clicks, locationdesc, topic, yearend):
    if n_clicks is not None:
        mortality_data = {'locationdesc': locationdesc, 'topic': topic, 'yearend': yearend}
        prediction_mor = predict_mortality_data(mortality_data, columns_order)
        prediction_mor_rf = predict_mortality_data_rf(mortality_data, columns_order)
        prediction_mor_xgb = predict_mortality_data_xgb(mortality_data, columns_order)
        prediction_mor_rid = predict_mortality_data_rid(mortality_data, columns_order)
         # Concatenate the results into a single string

        output_line1 = f'Linear Regression -> Mortality Rate: {int(prediction_mor)} & r2: {linear_r2_value:.4f}'
        output_line2 = f'Ridge Regression  -> Mortality Rate: {int(prediction_mor_rid)} & r2: {linear_rid_value:.4f}'
        output_line3 = f'Random Forest -> Mortality Rate: {int(prediction_mor_rf)} & r2: {linear_rf_value:.4f}'
        output_line4 = f'XGBoost -> Mortality Rate: {int(prediction_mor_xgb)} & r2: {linear_xgb_value:.4f}'

        result_string = f'{output_line1} ... {output_line2} ... {output_line3} ... {output_line4}'
        #Return the formatted string to update the 'prediction-output' component
        #return dcc.Markdown(result_string, style={'color': 'black', 'font-weight': 'bold', 'font-size': 24, 'white-space': 'pre-line'})
        #return dcc.Markdown(result_string)
        
        result_paragraphs = [
            html.P(output_line1),
            html.P(output_line2),
            html.P(output_line3),
            html.P(output_line4)
        ]

        # Return the formatted paragraphs to update the 'prediction-output' component
        return html.Div([
            #dcc.Markdown(result_string),
            html.Div(result_paragraphs)
        ])
    
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)

