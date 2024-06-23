
import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
from sklearn.preprocessing import LabelEncoder

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

    if st.button(f'Show {trend_type} Trend'):
        climate_trend(country_name, trend_type)

if __name__ == '__main__':
    main()
    

cleaned_data = pd.read_csv('Q2merged_rf_temp_crop data.csv')
def crop_temp_scatter(country_name, selected_crop):
    country_data = cleaned_data[cleaned_data['Name'] == country_name]
    crop_data = country_data[country_data['Item'] == selected_crop]

    plt.figure(figsize=(10, 6))
    ct_s = sns.scatterplot(x='Temperature', y='Crop Yield', data=crop_data)
    ct_s.set_title(f'Temperature vs. {selected_crop} Yield in {country_name}')
    sns.regplot(x='Temperature', y='Crop Yield', data=crop_data, scatter=False, ax=ct_s)
    st.pyplot(plt)  # Display the plot in Streamlit

def main():
    st.title('Crop Yield vs. Temperature Scatter Plot')
    st.subheader('Select a Country')

    # Country selection
    country_name = st.selectbox('Choose a country', cleaned_data['Name'].unique(), key="country_select")

    # Get available crops for the selected country
    if country_name:
        country_data = cleaned_data[cleaned_data['Name'] == country_name]
        crop_available = country_data['Item'].unique()

        st.write(f"Available crop types in {country_name}:")
        selected_crop = st.selectbox('Choose a crop', crop_available, key="crop_select")

        if st.button('Show Scatter Plot'):
            crop_temp_scatter(country_name, selected_crop)

if __name__ == '__main__':
    main()
    
#Question 3
st.subheader('Question 3')
st.markdown('Does an increase in pesticide use cause an increase in crop yields, and does this causal relationship differ by country? ')

st.subheader('Boxplots for Fertilizers Used and Crop Yield of Selected Country')

#Question 4

merged_cleaned_data = pd.read_csv('merged_cleaned_data.csv')

st.title('Crop Yield Prediction and Analysis')

# Data preprocessing: Encode the categorical 'Main Climate Zone'
encoder = LabelEncoder()
merged_cleaned_data['Main Climate Zone'] = encoder.fit_transform(merged_cleaned_data['Main Climate Zone'])

# Function to train and visualize linear regression
def train_linear_regression(X_train, y_train, X_test, y_test, df, crop_name, crop_yield):
    # Initialize Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Print Linear Regression Equation
    coef_lr = lr_model.coef_
    intercept_lr = lr_model.intercept_
    equation = f'{crop_name}_yield = {intercept_lr:.2f}'
    feature_names = X_train.columns
    for coef, feature in zip(coef_lr, feature_names):
        equation += f' + ({coef:.2f} * {feature})'
    st.write(f'Linear Regression Equation for {crop_name}: {equation}')

    # Evaluate Linear Regression Model
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    st.write(f'Linear Regression - MSE: {mse_lr}, MAE: {mae_lr}, R²: {r2_lr}')

    # Visualize the regression line
    features = ['Surface Air Temperature(°C)', 'Precipitation(mm)', 'Main Climate Zone', 'Pesticide Used(tn)']
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 20))
    for i, feature in enumerate(features):
        sns.regplot(x=feature, y=crop_yield, data=df,
                    scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ax=axes[i])
        axes[i].set_title(f'{crop_name.capitalize()} Yield vs {feature}')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Return model and predictions
    return lr_model, mse_lr, mae_lr, r2_lr

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
    model_df = copy.deepcopy(merged_cleaned_data[['Main Climate Zone','Pesticide Used(tn)', 
                                                  'Surface Air Temperature(°C)', 'Precipitation(mm)', 
                                                  crop_yield_column]].dropna(subset=[crop_yield_column]))
    
    X = model_df.drop(crop_yield_column, axis=1)
    y = model_df[crop_yield_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Train the model and display results
    if st.button('Train Linear Regression Model'):
        train_linear_regression(X_train, y_train, X_test, y_test, model_df, selected_crop_name, crop_yield_column)

if __name__ == '__main__':
    main()