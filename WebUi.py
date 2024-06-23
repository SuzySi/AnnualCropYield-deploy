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

import copy

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

# Filter out extreme outliers
mean_fertilizer_data = mean_fertilizer_data[mean_fertilizer_data['Mean Pesticide Use (tonnes)'] < 1e6]

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

    if st.button('Show Graph',key="rest_dt_anly"):
        plot_average(mean_fertilizer_data, regression_line)

if __name__ == '__main__':
    main()

#Question 4

# Load the cleaned data
merged_cleaned_data = pd.read_csv('merged_cleaned_data.csv')

# Data preprocessing: Encode the categorical 'Main Climate Zone'
encoder = LabelEncoder()
merged_cleaned_data['Main Climate Zone'] = encoder.fit_transform(merged_cleaned_data['Main Climate Zone'])

# Function to train and visualize linear regression
def train_linear_regression(X_train, y_train, X_test, y_test, df, crop_name, crop_yield):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    coef_lr = lr_model.coef_
    intercept_lr = lr_model.intercept_
    equation = f'{crop_name}_yield = {intercept_lr:.2f}'
    feature_names = X_train.columns
    for coef, feature in zip(coef_lr, feature_names):
        equation += f' + ({coef:.2f} * {feature})'
    st.write(f'Linear Regression Equation for {crop_name}: {equation}')
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    st.write(f'Linear Regression - MSE: {mse_lr}, MAE: {mae_lr}, R²: {r2_lr}')
    features = ['Surface Air Temperature(°C)', 'Precipitation(mm)', 'Main Climate Zone', 'Pesticide Used(tn)']
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 20))
    for i, feature in enumerate(features):
        sns.regplot(x=feature, y=crop_yield, data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ax=axes[i])
        axes[i].set_title(f'{crop_name.capitalize()} Yield vs {feature}')
    plt.tight_layout()
    st.pyplot(fig)
    return lr_model, mse_lr, mae_lr, r2_lr

# Function to train and evaluate Random Forest
def train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    st.write(f'Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R²: {r2_rf}')
    return rf_model, mse_rf, mae_rf, r2_rf

# Function to train and evaluate Polynomial Regression
def train_polynomial_regression(X_train, y_train, X_test, y_test, X, crop_name):
    mae_scores = []
    for deg in range(2, 10):
        poly_deg = PolynomialFeatures(degree=deg)
        X_train_poly = poly_deg.fit_transform(X_train)
        X_test_poly = poly_deg.transform(X_test)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)
        poly_mae = mean_absolute_error(y_test, y_pred_poly)
        mae_scores.append(poly_mae)
    poly_deg = np.arange(2, 10)
    plt.figure(figsize=(10, 6))
    plt.plot(poly_deg, mae_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Degree of Polynomial vs. MAE')
    plt.grid(True)
    st.pyplot(plt)
    optimal_poly_deg = poly_deg[np.argmin(mae_scores)]
    st.write(f'Optimal degree of polynomial for {crop_name}: {optimal_poly_deg}')
    poly_deg = PolynomialFeatures(degree=optimal_poly_deg)
    X_train_poly = poly_deg.fit_transform(X_train)
    X_test_poly = poly_deg.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    coef_poly = poly_model.coef_
    intercept_poly = poly_model.intercept_
    feature_names = X.columns
    equation = f'{crop_name}_yield = {intercept_poly:.2f}'
    for coef, feature in zip(coef_poly[1:], feature_names):
        equation += f' + ({coef:.2f} * {feature})'
    st.write(f'Polynomial Regression Equation for {crop_name}: {equation}')
    y_pred_poly = poly_model.predict(X_test_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    st.write(f'Polynomial Regression - MSE: {mse_poly}, MAE: {mae_poly}, R²: {r2_poly}')
    return poly_model, mse_poly, mae_poly, r2_poly

# Function to train and evaluate Gradient Boosting
def train_gradient_boosting(X_train, y_train, X_test, y_test):
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    st.write(f'Gradient Boosting - MSE: {mse_gb}, MAE: {mae_gb}, R²: {r2_gb}')
    return gb_model, mse_gb, mae_gb, r2_gb

# Function to train and evaluate Decision Tree
def train_decision_tree(X_train, y_train, X_test, y_test, crop_name):
    mae_scores = []
    for md in range(1, 21):
        dtr = DecisionTreeRegressor(max_depth=md)
        dtr.fit(X_train, y_train)
        y_pred_dtr = dtr.predict(X_test)
        dtr_mae = mean_absolute_error(y_test, y_pred_dtr)
        mae_scores.append(dtr_mae)
    max_depths = np.arange(1, 21)
    plt.figure(figsize=(8, 5))
    plt.plot(max_depths, mae_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'{crop_name.capitalize()} Decision Tree Max Depth vs. MAE')
    plt.grid(True)
    st.pyplot(plt)
    optimal_max_depth = max_depths[np.argmin(mae_scores)]
    st.write(f'Optimal max depth for {crop_name} Decision Tree: {optimal_max_depth}')
    dtr_model = DecisionTreeRegressor(max_depth=optimal_max_depth)
    dtr_model.fit(X_train, y_train)
    y_pred_dtr = dtr_model.predict(X_test)
    mse_dtr = mean_squared_error(y_test, y_pred_dtr)
    mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
    r2_dtr = r2_score(y_test, y_pred_dtr)
    st.write(f'Decision Tree - MSE: {mse_dtr}, MAE: {mae_dtr}, R²: {r2_dtr}')
    return dtr_model, mse_dtr, mae_dtr, r2_dtr

def main():
    st.subheader('Question 4: Predict Future Crop Yields')
    st.markdown('Predict future crop yields based on current and historical data on pesticide use, temperature, and precipitation.')

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
    crop_yield_column = crops[selected_crop_name]

    # Filter data for the selected crop
    model_df = merged_cleaned_data[['Main Climate Zone', 'Pesticide Used(tn)', 
                                    'Surface Air Temperature(°C)', 'Precipitation(mm)', 
                                    crop_yield_column]].dropna(subset=[crop_yield_column])
    
    X = model_df.drop(crop_yield_column, axis=1)
    y = model_df[crop_yield_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Model selection for training
    model_options = ['Linear Regression', 'Random Forest', 'Polynomial Regression', 'Gradient Boosting', 'Decision Tree']
    selected_model = st.selectbox('Choose a model to train', model_options)

    # Initialize rf_model
    rf_model = None

    if st.button('Train Model',key="show_train"):
        if selected_model == 'Linear Regression':
            train_linear_regression(X_train, y_train, X_test, y_test, model_df, selected_crop_name, crop_yield_column)
        elif selected_model == 'Random Forest':
            rf_model, _, _, _, = train_random_forest(X_train, y_train, X_test, y_test)
            st.session_state.rf_model = rf_model  # Store in session state
        elif selected_model == 'Polynomial Regression':
            train_polynomial_regression(X_train, y_train, X_test, y_test, X, selected_crop_name)
        elif selected_model == 'Gradient Boosting':
            train_gradient_boosting(X_train, y_train, X_test, y_test)
        elif selected_model == 'Decision Tree':
            train_decision_tree(X_train, y_train, X_test, y_test, selected_crop_name)

    # Add section for user input and prediction using Random Forest
    st.subheader('Predict Future Yield Using Random Forest')
    crop_type = st.selectbox('Select Crop Type', list(crops.keys()))
    climate_zone = st.selectbox('Select Main Climate Zone', encoder.classes_.tolist())
    climate_zone_encoded = encoder.transform([climate_zone])[0]
    pesticide_use = st.number_input('Pesticide Used (tn)', value=100.0)
    temperature = st.number_input('Surface Air Temperature (°C)', value=20.0)
    precipitation = st.number_input('Precipitation (mm)', value=50.0)

    if st.button('Predict Future Yield',key="show_model"):
        if 'rf_model' in st.session_state:
            user_input = np.array([[climate_zone_encoded, pesticide_use, temperature, precipitation]])
            predicted_yield = st.session_state.rf_model.predict(user_input)
            st.write(f'The predicted yield for {selected_crop_name} is: {predicted_yield[0]:.2f} hg/ha')
        else:
            st.write('Please train the Random Forest model first.')

if __name__ == '__main__':
    main()