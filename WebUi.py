import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

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
