import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import os


st.title('Analysis of Crop Yield of Different Countries from Year 1990 to Year 2016 ')
st.markdown('Crop yields contribute to an adequate food supply, reducing hunger, and improving farmers incomes, which in turn supports national economies. With climate change posing significant challenges, analysing and predicting crop yields helps develop resilient agricultural practices. Analysis on the influence of environmental factors and pesticide use influence on various crop yield across different countries over time is vital ')

st.subheader('Question 1')
# Load the cleaned pivot table CSV file into a DataFrame
cleaned_data = pd.read_csv('cleaned_average_annual_crop_yield.csv', index_col='Country')
st.subheader('Average Annual Crop Yield from Year 1990 to 2016 (hg/ha)')

country = st.selectbox('Select Country', cleaned_data.index.tolist())
year = st.selectbox('Select Year', cleaned_data.columns.tolist())
if country and year:
    avg_yield = cleaned_data.at[country, year]
    st.write(f"The average annual crop yield for {country} in {year} is {avg_yield:.2f}")


# Line Plot
st.subheader('Line Plot of Average Annual Crop Yield for Selected Countries from Year 1990 to Year 2016')

selected_countries = st.multiselect('Select countries', cleaned_data.index.tolist(), default=['United States of America', 'China', 'India', 'Brazil', 'Australia'])

if selected_countries:
    data_to_plot = cleaned_data.loc[selected_countries].T

    plt.figure(figsize=(14, 7))
    for country in selected_countries:
        plt.plot(data_to_plot.index, data_to_plot[country], label=country)
    plt.xlabel('Year')
    plt.ylabel('Average Yield(hg/ha)')
    plt.title('Average Annual Crop Yield (1990-2016)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Bar Plot
st.subheader('Bar Plot of Average Crop Yield for a Specific Year')

# Add a multiselect widget for selecting countries
selected_country = st.multiselect('Select countries', cleaned_data.index.tolist(), default=['United States of America', 'China', 'India', 'Brazil', 'Australia'],key='country_selection')

year = st.selectbox('Select year', cleaned_data.columns.tolist(), index=cleaned_data.columns.tolist().index('2016'))

if year:
    # Filter data for selected countries
    data_for_year = cleaned_data.loc[selected_country, year].dropna().sort_values(ascending=False).head(10)

    plt.figure(figsize=(14, 7))
    sns.barplot(x=data_for_year.values, y=data_for_year.index, palette='viridis')
    plt.xlabel('Average Yield (hg/ha)')
    plt.ylabel('Country')
    plt.title(f'Average Crop Yield in {year} for Selected Countries')
    st.pyplot(plt)

#Question 2
filtered_rftemp = pd.read_csv('Q2merged_rf_temp_crop data.csv')

def climate_trend(country_name, trend_type):
    country_data = filtered_rftemp[filtered_rftemp['Name'] == country_name]

    plt.figure(figsize=(10, 6))
    if trend_type == 'Precipitation':
        plt.plot(country_data['Year'], country_data['Precipitation'], marker='o')
        plt.title(f'Precipitation Trend for {country_name} (1990-2016)')
        plt.ylabel('Precipitation')
    elif trend_type == 'Temperature':
        plt.plot(country_data['Year'], country_data['Temperature'], marker='o')
        plt.title(f'Temperature Trend for {country_name} (1990-2016)')
        plt.ylabel('Temperature')

    plt.xlabel('Year')
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    st.subheader('Question 2')
    st.markdown('How do trends in temperature and precipitation vary among different countries and over different years(1990-2016), and how do they effect the crop yield?')
    st.subheader('Climate Trends Visualization')
    

    country_name = st.selectbox('Choose a country', filtered_rftemp['Name'].unique())
    trend_type = st.radio('Select trend type:', ['Precipitation', 'Temperature'])

    if st.button(f'Show {trend_type} Trend',key="show_trend"):
        climate_trend(country_name, trend_type)

if __name__ == '__main__':
    main()
    

# Load the data
q2cleaned_data = pd.read_csv('Q2merged_rf_temp_crop data.csv')

def crop_temp_scatter(country_name, selected_crop):
    country_data = q2cleaned_data[q2cleaned_data['Name'] == country_name]
    crop_data = country_data[country_data['Item'] == selected_crop]

    plt.figure(figsize=(10, 6))
    ct_s = sns.scatterplot(x='Temperature', y='Crop Yield', data=crop_data)
    ct_s.set_title(f'Temperature vs. {selected_crop} Yield in {country_name}')
    sns.regplot(x='Temperature', y='Crop Yield', data=crop_data, scatter=False, ax=ct_s)
    st.pyplot(plt)  # Display the plot in Streamlit

def crop_rf_scatter(country_name, selected_crop):
    country_data = q2cleaned_data[q2cleaned_data['Name'] == country_name]
    crop_data = country_data[country_data['Item'] == selected_crop]

    plt.figure(figsize=(10, 6))
    crf_s = sns.scatterplot(x='Precipitation', y='Crop Yield', data=crop_data)
    crf_s.set_title(f'Precipitation vs. {selected_crop} Yield in {country_name}')
    sns.regplot(x='Precipitation', y='Crop Yield', data=crop_data, scatter=False, ax=crf_s)
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    st.subheader('Crop Yield Scatter Plots')
    
    # Temperature vs. Crop Yield Section
    st.subheader('Temperature vs. Crop Yield Scatter Plot')
    country_name_temp = st.selectbox('Choose a country for temperature scatter plot', q2cleaned_data['Name'].unique(), key="c_select_temp")
    if country_name_temp:
        country_data_temp = q2cleaned_data[q2cleaned_data['Name'] == country_name_temp]
        crop_available_temp = country_data_temp['Item'].unique()
        selected_crop_temp = st.selectbox('Choose a crop for temperature scatter plot', crop_available_temp, key="crop_select_temp")
        if st.button('Show Temperature vs. Crop Yield Scatter Plot', key="show_scatter_temp"):
            crop_temp_scatter(country_name_temp, selected_crop_temp)

    # Precipitation vs. Crop Yield Section
    st.subheader('Precipitation vs. Crop Yield Scatter Plot')
    country_name_prec = st.selectbox('Choose a country for precipitation scatter plot', q2cleaned_data['Name'].unique(), key="c_select_prec")
    if country_name_prec:
        country_data_prec = q2cleaned_data[q2cleaned_data['Name'] == country_name_prec]
        crop_available_prec = country_data_prec['Item'].unique()
        selected_crop_prec = st.selectbox('Choose a crop for precipitation scatter plot', crop_available_prec, key="crop_select_prec")
        if st.button('Show Precipitation vs. Crop Yield Scatter Plot', key="show_scatter_prec"):
            crop_rf_scatter(country_name_prec, selected_crop_prec)

if __name__ == '__main__':
    main()
    
    
#Question 3
st.subheader('Question 3')
st.markdown('Does an increase in pesticide use cause an increase in crop yields, and does this causal relationship differ by country? ')


# Load the dataset
data = pd.read_csv('cleaned_pesticide_data.csv')

# Function to plot the graph for a specific area in pesticide dataset
def plot_pesticide_data(area_name, data):
    if area_name not in data['Area'].unique():
        st.error(f"Area '{area_name}' not found in the dataset.")
        return
    
    area_data = data[data['Area'] == area_name]
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    
    # Boxplot for Pesticide Use
    sns.boxplot(ax=axes[0], x=area_data['Pesticide Use (tonnes)'])
    axes[0].set_title(f'Pesticide Use in {area_name}')
    axes[0].set_xlabel('Pesticide Use (tonnes)')
    
    # Boxplot for Crop Yield
    sns.boxplot(ax=axes[1], x=area_data['Crop Yield (hg/ha)'])
    axes[1].set_title(f'Crop Yield in {area_name}')
    axes[1].set_xlabel('Crop Yield (hg/ha)')
    
    # Scatter plot for the relationship between pesticide use and crop yields
    sns.scatterplot(ax=axes[2], x='Pesticide Use (tonnes)', y='Crop Yield (hg/ha)', data=area_data)
    axes[2].set_title(f'Relationship between Pesticide Use and Crop Yields in {area_name}')
    axes[2].set_xlabel('Pesticide Use (tonnes)')
    axes[2].set_ylabel('Crop Yield (hg/ha)')
    
    # Linear regression
    X = area_data['Pesticide Use (tonnes)'].values.reshape(-1, 1)
    Y = area_data['Crop Yield (hg/ha)'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)
    
    axes[2].plot(X, Y_pred, color='red', linewidth=2, label='Linear regression')
    axes[2].legend()
    
    st.pyplot(fig)

def main():
    st.title('Pesticide Use vs. Crop Yield Analysis')
    st.subheader('Select an Area (Country)')

    # Country selection
    area_name = st.selectbox('Choose an area (country)', data['Area'].unique())

    if st.button('Show Analysis'):
        plot_pesticide_data(area_name, data)

if __name__ == '__main__':
    main()

# Load the dataset
data = pd.read_csv('cleaned_fertilizer_data.csv')

def plot_fertilizer_data(country_name, data):
    if country_name not in data['Country'].unique():
        st.write(f"Country '{country_name}' not found in the dataset.")
        return
    
    country_data = data[data['Country'] == country_name]
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    
    # Boxplot for Fertilizer Use
    sns.boxplot(ax=axes[0], x=country_data['Nitrogen Fertilizer Use (kg/ha)'])
    axes[0].set_title(f'Fertilizer Use in {country_name}')
    axes[0].set_xlabel('Fertilizer Use (kg/ha)')
    
    # Boxplot for Crop Yield
    sns.boxplot(ax=axes[1], x=country_data['Cereal Yield (tonnes/ha)'])
    axes[1].set_title(f'Crop Yield in {country_name}')
    axes[1].set_xlabel('Crop Yield (tonnes/ha)')
    
    st.pyplot(fig)  # Display the plot in Streamlit
    
    # Scatter plot for the relationship between fertilizer use and crop yields
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(country_data['Nitrogen Fertilizer Use (kg/ha)'], country_data['Cereal Yield (tonnes/ha)'], label='Data points')
    ax.set_title(f'Relationship between Fertilizer Use and Crop Yields in {country_name}')
    ax.set_xlabel('Fertilizer Use (kg/ha)')
    ax.set_ylabel('Crop Yield (tonnes/ha)')
    
    # Linear regression
    X = country_data['Nitrogen Fertilizer Use (kg/ha)'].values.reshape(-1, 1)
    Y = country_data['Cereal Yield (tonnes/ha)'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)
    
    ax.plot(X, Y_pred, color='red', linewidth=2, label='Linear regression')
    ax.legend()
    
    st.pyplot(fig)  # Display the plot in Streamlit

def main():
    st.subheader('Fertiliser vs Crop Yield For A Specific Country')
    st.subheader('Select a Country')

    # Country selection
    country_name = st.selectbox('Choose a country', data['Country'].unique())

    if st.button('Show Graph'):
        plot_fertilizer_data(country_name, data)

if __name__ == '__main__':
    main()

# Load the dataset
fertilizer_data = pd.read_csv('cleaned_pesticide_data.csv')

# Calculate mean values for each country
def calculate_mean_values(data):
    countries = data['Area'].unique()
    mean_values = []
    for country in countries:
        country_data = data[data['Area'] == country]
       
        mean_pesticide_use = country_data['Pesticide Use (tonnes)'].mean()
        mean_crop_yield = country_data['Crop Yield (hg/ha)'].mean()
        mean_values.append({
            'Country': country,
            'Mean Pesticide Use (tonnes)': mean_pesticide_use,
            'Mean Crop Yield (hg/ha)': mean_crop_yield
        })
    mean_data = pd.DataFrame(mean_values)
    return mean_data

mean_fertilizer_data = calculate_mean_values(fertilizer_data)

# Fit linear regression
X = mean_fertilizer_data[['Mean Pesticide Use (tonnes)']]
y = mean_fertilizer_data['Mean Crop Yield (hg/ha)']
model = LinearRegression()
model.fit(X, y)
regression_line = model.predict(X)

# Function to plot the graph with linear regression line
def plot_average(data, regression_line):
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(data['Mean Pesticide Use (tonnes)'], data['Mean Crop Yield (hg/ha)'], alpha=0.7, label='Data Points')
    
    # Regression line
    plt.plot(data['Mean Pesticide Use (tonnes)'], regression_line, color='red', label='Regression Line')
    
    plt.title('Relationship between Average Pesticide Use by Country and Mean Crop Yield')
    plt.xlabel('Average Pesticide Use (tonnes)')
    plt.ylabel('Mean Crop Yield (hg/ha)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    st.subheader('Pesticide Data Analysis')
    st.subheader('Relationship between Average Pesticide Use by Country and Mean Crop Yield')

    if st.button('Show Graph',key="pest_country_mean"):
        plot_average(mean_fertilizer_data, regression_line)

if __name__ == '__main__':
    main()

# # Load the dataset
# fertilizer_data = pd.read_csv('cleaned_pesticide_data.csv')

# # Calculate mean values for each country
# def calculate_mean_values(data):
#     countries = data['Area'].unique()
#     mean_values = []
#     for country in countries:
#         country_data = data[data['Area'] == country]
       
#         mean_pesticide_use = country_data['Pesticide Use (tonnes)'].mean()
#         mean_crop_yield = country_data['Crop Yield (hg/ha)'].mean()
#         mean_values.append({
#             'Country': country,
#             'Mean Pesticide Use (tonnes)': mean_pesticide_use,
#             'Mean Crop Yield (hg/ha)': mean_crop_yield
#         })
#     mean_data = pd.DataFrame(mean_values)
#     return mean_data

# mean_fertilizer_data = calculate_mean_values(fertilizer_data)

# # Filter out extreme outliers
# mean_fertilizer_data = mean_fertilizer_data[mean_fertilizer_data['Mean Pesticide Use (tonnes)'] < 1e6]

# # Fit linear regression
# X = mean_fertilizer_data[['Mean Pesticide Use (tonnes)']]
# y = mean_fertilizer_data['Mean Crop Yield (hg/ha)']
# model = LinearRegression()
# model.fit(X, y)
# regression_line = model.predict(X)

# # Function to plot the graph with linear regression line
# def plot_average(data, regression_line):
#     plt.figure(figsize=(12, 8))
    
#     # Scatter plot
#     plt.scatter(data['Mean Pesticide Use (tonnes)'], data['Mean Crop Yield (hg/ha)'], alpha=0.7, label='Data Points')
    
#     # Regression line
#     plt.plot(data['Mean Pesticide Use (tonnes)'], regression_line, color='red', label='Regression Line')
    
#     plt.title('Relationship between Average Pesticide Use by Country and Mean Crop Yield')
#     plt.xlabel('Average Pesticide Use (tonnes)')
#     plt.ylabel('Mean Crop Yield (hg/ha)')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)  # Display the plot in Streamlit

# def main():
#     st.subheader('Pesticide Data Analysis')
#     st.subheader('Relationship between Average Pesticide Use by Country and Mean Crop Yield')
    
#     if st.button('Show Graph',key="rest_dt_anly"):
#         plot_average(mean_fertilizer_data, regression_line)

# if __name__ == '__main__':
#     main()

# Load the dataset
fertilizer_data = pd.read_csv('cleaned_pesticide_data.csv')

# Calculate mean values for each country
def calculate_mean_values(data):
    countries = data['Area'].unique()
    mean_values = []
    for country in countries:
        country_data = data[data['Area'] == country]
       
        mean_pesticide_use = country_data['Pesticide Use (tonnes)'].mean()
        mean_crop_yield = country_data['Crop Yield (hg/ha)'].mean()
        mean_values.append({
            'Country': country,
            'Mean Pesticide Use (tonnes)': mean_pesticide_use,
            'Mean Crop Yield (hg/ha)': mean_crop_yield
        })
    mean_data = pd.DataFrame(mean_values)
    return mean_data

mean_fertilizer_data = calculate_mean_values(fertilizer_data)

# Further filter out extreme outliers
mean_fertilizer_data = mean_fertilizer_data[mean_fertilizer_data['Mean Pesticide Use (tonnes)'] < 40000]

# Fit linear regression
X = mean_fertilizer_data[['Mean Pesticide Use (tonnes)']]
y = mean_fertilizer_data['Mean Crop Yield (hg/ha)']
model = LinearRegression()
model.fit(X, y)
regression_line = model.predict(X)

# Function to plot the graph with linear regression line
def plot_average(data, regression_line):
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.scatter(data['Mean Pesticide Use (tonnes)'], data['Mean Crop Yield (hg/ha)'], alpha=0.7, label='Data Points')
    
    # Regression line
    plt.plot(data['Mean Pesticide Use (tonnes)'], regression_line, color='red', label='Regression Line')
    
    plt.title('Relationship between Average Pesticide Use by Country and Mean Crop Yield')
    plt.xlabel('Average Pesticide Use (tonnes)')
    plt.ylabel('Mean Crop Yield (hg/ha)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    #st.subheader('Pesticide Data Analysis')
    st.subheader('Relationship between Average Pesticide Use by Country and Mean Crop Yield')
    st.markdown('Clearer Vision without outliers')


    if st.button('Show Graph',key="q3"):
        plot_average(mean_fertilizer_data, regression_line)

if __name__ == '__main__':
    main()

# Load the datasets
fertilizer_data = pd.read_csv('cleaned_fertilizer_data.csv')

# Calculate mean values for each country 
def calculate_mean_values(data):
    countries = data['Country'].unique()
    mean_values = []
    for country in countries:
        country_data = data[data['Country'] == country]
       
        mean_fertilizer_use = country_data['Nitrogen Fertilizer Use (kg/ha)'].mean()
        mean_crop_yield = country_data['Cereal Yield (tonnes/ha)'].mean()
        mean_values.append({
            'Country': country,
            'Mean Fertilizer Use (tonnes)': mean_fertilizer_use,
            'Mean Crop Yield (tonnes/ha)': mean_crop_yield
        })
    mean_data = pd.DataFrame(mean_values)
    return mean_data

mean_fertilizer_data = calculate_mean_values(fertilizer_data)

# Perform linear regression
X = mean_fertilizer_data[['Mean Fertilizer Use (tonnes)']]
y = mean_fertilizer_data['Mean Crop Yield (tonnes/ha)']

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Predictions for the regression line
y_pred = linear_regressor.predict(X)

# Function to plot the graph 
def plot_average(data, predictions):
    plt.figure(figsize=(12, 8))
    
    # Scatter plot 
    plt.scatter(data['Mean Fertilizer Use (tonnes)'], data['Mean Crop Yield (tonnes/ha)'], alpha=0.7, label='Data Points')
    
    # Regression line
    plt.plot(data['Mean Fertilizer Use (tonnes)'], predictions, color='red', linewidth=2, label='Regression Line')
    
    # Adding labels for each country
    #for i, row in data.iterrows():
        #plt.text(row['Mean Fertilizer Use (tonnes)'], row['Mean Crop Yield (tonnes/ha)'], row['Country'], fontsize=9)
    
    plt.title('Relationship between Average Nitrogen Use by Country and Mean Crop Yields for All Countries')
    plt.xlabel('Average Nitrogen Fertilizer Use (kg/ha)')
    plt.ylabel('Mean Crop Yield (tonnes/ha)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    st.subheader('Fertilizer Data Analysis')
    st.subheader('Relationship between Average Nitrogen Use by Country and Mean Crop Yields for All Countries')

    if st.button('Show Graph',key="q3_grp"):
        plot_average(mean_fertilizer_data, y_pred)

if __name__ == '__main__':
    main()

#Question 4

def main():
    st.subheader('Question 4')
    merged_cleaned_data = pd.read_csv('merged_cleaned_data.csv')
    encoder = LabelEncoder()
    encoder.fit(merged_cleaned_data['Main Climate Zone'])

     # Crop selection for modeling
    crops = {
        'Pulses': 'pulses_yield',
        'Maize': 'maize_yield',
        'Sugar Crops': 'sugar_crops_yield',
        'Tobacco': 'tobacco_yield',
        'Rice': 'rice_yield',
        'Wheat': 'wheat_yield'
    }
    selected_crop_name = st.selectbox('Choose a crop for modeling', list(crops.keys()))
    #crop_yield_column = crops[selected_crop_name]


    # Get a list of all joblib files in the models folder
    models_folder = 'models'
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.joblib')]

    # Add a selectbox to choose the model
    st.subheader('Select a Trained Model')
    selected_model_file = st.selectbox('Choose a model', model_files)

    # Load the selected model from joblib file
    model_file_path = os.path.join(models_folder, selected_model_file)
    loaded_model = joblib.load(model_file_path)

    # Add section for user input and prediction using loaded model
    st.subheader('Predict Future Yield Using Loaded Model')
    crop_type = st.selectbox('Select Crop Type', list(crops.keys()))
    climate_zone = st.selectbox('Select Main Climate Zone', encoder.classes_.tolist())
    climate_zone_encoded = encoder.transform([climate_zone])[0]
    pesticide_use = st.number_input('Pesticide Used (tn)', value=100.0)
    temperature = st.number_input('Surface Air Temperature (°C)', value=20.0)
    precipitation = st.number_input('Precipitation (mm)', value=50.0)

    if st.button('Predict Future Yield',key="show_model"):
        user_input = np.array([[climate_zone_encoded, pesticide_use, temperature, precipitation]])
        predicted_yield = loaded_model.predict(user_input)
        st.write(f'The predicted yield for {selected_crop_name} is: {predicted_yield[0]:.2f} hg/ha')

if __name__ == '__main__':
    main()