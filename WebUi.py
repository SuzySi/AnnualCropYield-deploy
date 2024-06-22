import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Load the cleaned pivot table CSV file into a DataFrame
cleaned_data = pd.read_csv('cleaned_average_annual_crop_yield.csv', index_col='Country')
st.title('Average Annual Crop Yield from Year 1990 to 2016')

country = st.selectbox('Select Country', cleaned_data.index.tolist())
year = st.selectbox('Select Year', cleaned_data.columns.tolist())
if country and year:
    avg_yield = cleaned_data.at[country, year]
    st.write(f"The average annual crop yield for {country} in {year} is {avg_yield:.2f}")

year_for_top5 = st.selectbox('Select year to display top 5 countries', cleaned_data.columns.tolist(), key='top5_year')
if year_for_top5:
    top5_data = cleaned_data[year_for_top5].dropna().sort_values(ascending=False).head(5)
plt.figure(figsize=(14, 7))
sns.barplot(x=top5_data.values, y=top5_data.index, palette='viridis')
plt.xlabel('Average Yield')
plt.ylabel('Country')
plt.title(f'Top 5 Countries by Crop Yield in {year_for_top5}')
st.pyplot(plt)




# Line Plot
st.subheader('Line Plot of Average Annual Crop Yield for Selected Countries')

selected_countries = st.multiselect('Select countries', cleaned_data.index.tolist(), default=['United States of America', 'China', 'India', 'Brazil', 'Australia'])

if selected_countries:
    data_to_plot = cleaned_data.loc[selected_countries].T

    plt.figure(figsize=(14, 7))
    for country in selected_countries:
        plt.plot(data_to_plot.index, data_to_plot[country], label=country)
    plt.xlabel('Year')
    plt.ylabel('Average Yield')
    plt.title('Average Annual Crop Yield (1990-2016)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Heatmap
st.subheader('Heatmap of Average Annual Crop Yield')

plt.figure(figsize=(20, 10))
sns.heatmap(cleaned_data, cmap='YlGnBu', cbar=True)
plt.xlabel('Year')
plt.ylabel('Country')
plt.title('Heatmap of Average Annual Crop Yield (1990-2016)')
st.pyplot(plt)

# Bar Plot
st.subheader('Bar Plot of Average Crop Yield for a Specific Year')

year = st.selectbox('Select year', cleaned_data.columns.tolist(), index=cleaned_data.columns.tolist().index('2016'))

if year:
    data_for_year = cleaned_data[year].dropna().sort_values(ascending=False).head(10)

    plt.figure(figsize=(14, 7))
    sns.barplot(x=data_for_year.values, y=data_for_year.index, palette='viridis')
    plt.xlabel('Average Yield')
    plt.ylabel('Country')
    plt.title(f'Average Crop Yield in {year}')
    st.pyplot(plt)

# Scatter Plot
st.subheader('Scatter Plot of Average Annual Crop Yield')

long_data = cleaned_data.reset_index().melt(id_vars='Country', var_name='Year', value_name='Yield')

plt.figure(figsize=(14, 7))
sns.scatterplot(x='Year', y='Yield', hue='Country', data=long_data, legend=False)
plt.xlabel('Year')
plt.ylabel('Average Yield')
plt.title('Scatter Plot of Average Annual Crop Yield (1990-2016)')
st.pyplot(plt)

