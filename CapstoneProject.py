"""
Nicole Ciaccia 

Data Science Academy Capstone Project

January 2022
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(layout='wide')

# Title for dashboard
st.title("Welcome to my Data Science Dashboard!")
st.markdown('### This dashboard examines several questions about the provided data to see \
             if there are any apparent relationships between variables')
st.markdown('By: Nicole Ciaccia - PAR PEP team - January 2022')


# Read in data
df = pd.read_csv("data_capstone_dsa2021_2022.csv")

# Cleaning up state column. First uppercase all for easier standardization
df['state'] = df['state'].str.upper()

# Create a dictionary  of all state names and abbreviations
us_state_to_abbrev = {
	    "ALABAMA": "AL",
	    "ALASKA": "AK",
	    "ARIZONA": "AZ",
	    "ARKANSAS": "AR",
	    "CALIFORNIA": "CA",
	    "COLORADO": "CO",
	    "CONNECTICUT": "CT",
	    "DELAWARE": "DE",
	    "FLORIDA": "FL",
	    "GEORGIA": "GA",
	    "HAWAII": "HI",
	    "IDAHO": "ID",
	    "ILLINOIS": "IL",
	    "INDIANA": "IN",
	    "IOWA": "IA",
	    "KANSAS": "KS",
	    "KENTUCKY": "KY",
	    "LOUISIANA": "LA",
	    "MAINE": "ME",
	    "MARYLAND": "MD",
	    "MASSACHUSETTS": "MA",
	    "MICHIGAN": "MI",
	    "MINNESOTA": "MN",
	    "MISSISSIPPI": "MS",
	    "MISSOURI": "MO",
	    "MONTANA": "MT",
	    "NEBRASKA": "NE",
	    "NEVADA": "NV",
	    "NEW HAMPSHIRE": "NH",
	    "NEW JERSEY": "NJ",
	    "NEW MEXICO": "NM",
	    "NEW YORK": "NY",
	    "NORTH CAROLINA": "NC",
	    "NORTH DAKOTA": "ND",
	    "OHIO": "OH",
	    "OKLAHOMA": "OK",
	    "OREGON": "OR",
	    "PENNSYLVANIA": "PA",
	    "RHODE ISLAND": "RI",
	    "SOUTH CAROLINA": "SC",
	    "SOUTH DAKOTA": "SD",
	    "TENNESSEE": "TN",
	    "TEXAS": "TX",
	    "UTAH": "UT",
	    "VERMONT": "VT",
	    "VIRGINIA": "VA",
	    "WASHINGTON": "WA",
	    "WEST VIRGINIA": "WV",
	    "WISCONSIN": "WI",
	    "WYOMING": "WY",
        "PUERTO RICO": "PR",
	}

# get rid of special characters to start the cleaning
remove_characters = [".", ",", "(", ")"]

for character in remove_characters:
    df['state'] = df['state'].str.replace(character, "", regex=True)


# get rid of USA and UNITED STATES and turn slash and double space into single space
char_to_replace = {'/': ' ',
                   ' USA': '',
                   'USA ': '',
                   'USA': '',
                   ' UNITED STATES OF AMERICA': '',
                   ' UNITED STATES': '',
                   'UNITED STATES ': '',
                   ' US': '',
                   '  ': ' ',
                   }

for key, value in char_to_replace.items():
    df['state'] = df['state'].str.replace(key, value)



# Get abbreviations for already clean data (ie contains only state name)
df['temp1'] = df['state'].map(us_state_to_abbrev)
df = df.replace(np.nan, '', regex=True)


# Find any observations where the state listed is already the abrreviation
abbrevs= list(us_state_to_abbrev.values()) # turn values from state/abrrev dictionary into a list  
df['temp2'] = np.where(df['state'].isin(abbrevs), df['state'], "")


# combine anywhere where we now have state so far
df['state2']= df["temp1"] + df["temp2"]
df.drop(columns=['temp1', 'temp2'], inplace=True)


# Separate data into that which mapped successfully already and that which did not
df_good_state = df.loc[df['state2'] != ""]
df_no_state = df.loc[df['state2'] == ""]
df_no_state.drop(columns=['state2'], inplace=True) # since blank, dont need it and will get in the way later


# Look for observations where the state or abbrev is present within the string but may contain other words as well.
# Split state up by each individual word 
df_no_state_expand= df_no_state['state'].str.split(' ', expand=True)
# Rename columns so can use them later 
df_no_state_expand.columns = ['var1', 'var2', 'var3','var4', 'var5', 'var6','var7', 'var8', 'var9','var10', 'var11']


state_list= list(us_state_to_abbrev.keys()) # turn keys from state/abrrev dictionary into a list   

# Compare var1-11 to the state and abbrev lists and output anywhere that they match
for i in range(1,12):
    df_no_state_expand['find_state%s'%i] = np.where(df_no_state_expand['var%s'%i].isin(state_list), df_no_state_expand['var%s'%i], "")
    df_no_state_expand['find_abbrev%s'%i] = np.where(df_no_state_expand['var%s'%i].isin(abbrevs), df_no_state_expand['var%s'%i], "")
    
# collapse all of the find_state and Find_abbrev vars     
df_no_state_expand['state2_temp']=df_no_state_expand["find_state1"] + df_no_state_expand["find_state2"] \
    + df_no_state_expand["find_state3"] + df_no_state_expand["find_state4"] + df_no_state_expand["find_state5"] \
    + df_no_state_expand["find_state6"] + df_no_state_expand["find_state7"] + df_no_state_expand["find_state8"] \
    + df_no_state_expand["find_state9"] + df_no_state_expand["find_state10"] + df_no_state_expand["find_state11"]     

df_no_state_expand['abbrev_temp']=df_no_state_expand["find_abbrev1"] + df_no_state_expand["find_abbrev2"] \
    + df_no_state_expand["find_abbrev3"] + df_no_state_expand["find_abbrev4"] + df_no_state_expand["find_abbrev5"] \
    + df_no_state_expand["find_abbrev6"] + df_no_state_expand["find_abbrev7"] + df_no_state_expand["find_abbrev8"] \
    + df_no_state_expand["find_abbrev9"] + df_no_state_expand["find_abbrev10"] + df_no_state_expand["find_abbrev11"]     

    
# map those with full state name to original state dictionary to get the abbreviation 
df_no_state_expand['temp3'] = df_no_state_expand['state2_temp'].map(us_state_to_abbrev)
df_no_state_expand = df_no_state_expand.replace(np.nan, '', regex=True)

# combine anywhere that we now have state 
df_no_state_expand['state2'] = df_no_state_expand['abbrev_temp']+ df_no_state_expand['temp3']
df_no_state_expand=df_no_state_expand[['state2']] # only keep this var bc it is the only one we care about

#merge the new variable in with the original no_state dataframe (can merge on index solely)
df_no_state=df_no_state.merge(df_no_state_expand,left_index=True, right_index=True)

#combine the new no_state data that mostly has state2 filled in with the good_state data to get the final dataset. 
final_dataset= pd.concat([df_good_state, df_no_state])

final_dataset['state_final'] = np.where(final_dataset['state2'].isin(abbrevs), final_dataset['state2'], "Unknown_State")

# (end up with 92 observations with no state2, but that is the best we can do)
check_state = final_dataset.loc[final_dataset['state_final'] == "Unknown_State"]



st.text("")
st.text("")
st.markdown('##### Investigation 1: Do any individual items have a high correlation with the Total Score?')

# Get dataframe of just the scores
final_dataset2=final_dataset.iloc[:, 20:] 
final_dataset2.drop(columns=['state2', 'gender', 'home_computer', 'state', 'age', 'state_final'], inplace=True) 

# Run Pearson correlation
corrs=final_dataset2.corr(method ='pearson')

# Only keep the last row - correlation between item and total score
corrs2=corrs.loc[['sum_score']]

#get rid of sum_score column since it is unneeded
corrs2.drop(columns=['sum_score'], inplace=True)

#get rid of prefix in col names and transpose for easier graphing
corrs2.columns = corrs2.columns.str.strip('gs_')
corrs3 = corrs2.T
corrs3.reset_index(inplace=True)
corrs3.rename(columns={'index':'Item', 'sum_score':'correlation'}, inplace=True)

# Interactive bar chart of correlations for each item
fig1=px.bar(corrs3,x='Item', y='correlation')
st.plotly_chart(fig1)
st.markdown("**Findings:** No items jump out as showing a particularly high correlation with the total score.\
            Items 7 and 4 do the best job, but at only slightly above 0.5, that isn't all that high. \
            Item 20 clearly has the worst correlation with total score, but that could be due to its \
            position at the end of the test. No exciting conclusions here! ")
st.text("")


# Create Age Groups to see impact of Age on Score and Testing Time
bins= [0,20,30,40,50,60,110]
labels = ['Under 20','20-29','30-39','40-49', '50-59','60+']
final_dataset['AgeGroup'] = pd.cut(final_dataset['age'], bins=bins, labels=labels, right=False)

st.markdown('##### Investigation 2: Does Age Impact Total Score?')
fig_hist = px.histogram(final_dataset.loc[final_dataset.AgeGroup=='Under 20'],x='sum_score', title='Under 20')
fig_hist2 = px.histogram(final_dataset.loc[final_dataset.AgeGroup=='20-29'],x='sum_score', title='20-29')
fig_hist3 = px.histogram(final_dataset.loc[final_dataset.AgeGroup=='30-39'],x='sum_score', title='30-39')
fig_hist4 = px.histogram(final_dataset.loc[final_dataset.AgeGroup=='40-49'],x='sum_score', title='40-49')
fig_hist5= px.histogram(final_dataset.loc[final_dataset.AgeGroup=='50-59'],x='sum_score', title='50-59')
fig_hist6 = px.histogram(final_dataset.loc[final_dataset.AgeGroup=='60+'],x='sum_score', title='60+')

col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(fig_hist)  
    st.plotly_chart(fig_hist4)  
with col2:
    st.plotly_chart(fig_hist2)
    st.plotly_chart(fig_hist5)
with col3:
    st.plotly_chart(fig_hist3)
    st.plotly_chart(fig_hist6)




# st.markdown("**Findings:** Distributions for each age category seem to be relatively simlilar\
#             with most candidates in each age group getting a score of 19. \
#             We can also see from this histogram that the data is very left skewed. \
#             It appears that whatever this exam was, the questions were not difficult.")




# names = labels
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
# final_dataset.hist('sum_score', ax=axes[0,0])

# plt.show()



# for ax, name in zip(axes, names):
#     ax.set(xticks=[], yticks=[], title=name)





# rwgor.loc[rwgor.attractionName >=165
















# Histogram of testing time 
# bins_list = [0, 300, 600, 900, 1200, 1500, 3000, 4600]
# fig2, ax = plt.subplots()
# plt.hist(final_dataset.rt_total,edgecolor="black", color="m", bins=bins_list)
# plt.xticks(bins_list, rotation = 90)
# plt.title("Histogram of Total Testing Time")
# plt.xlabel("Total Testing Time")
# plt.ylabel("Frequencies")
# st.pyplot(fig2)

# Interactive Bar chart of Mean Scores by State and Gender
state_gender_means = final_dataset.groupby(['gender', 'state_final'])['sum_score'].mean().reset_index()
f_means = state_gender_means.loc[state_gender_means['gender'] == 'Female']
m_means = state_gender_means.loc[state_gender_means['gender'] == 'Male']
state_gender_means2=f_means.append(m_means)
state_gender_means2.sort_values(by=['state_final'], inplace=True)

states=st.selectbox('State:', abbrevs) #create selector box
state_gender_means3=state_gender_means2.query('state_final==@states') #query the data using the selector box
test=px.bar(state_gender_means3,x='gender',y='sum_score', color='gender', title ="Mean Scores by Gender and State")
st.plotly_chart(test)







fig4 = px.bar(state_gender_means2, x="state_final", color="gender",
             y='sum_score',
             title="A Grouped Bar Chart With Plotly Express in Python",
             barmode='group',
             height=700,
             width=1000,
             facet_row="gender"
            )

st.plotly_chart(fig4)




# col1, col2, col3, col4, col5 =st.columns((2,2,5,1,5))

# for i in range(1,11):
#     chk01 = col1.checkbox('Item %s'%i, value = 1 )
    
# for i in range(11,21):
#     chk01 = col2.checkbox('Item %s'%i, value = 1 )




from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=3, cols=3, subplot_titles=labels)
#fig.add_trace(go.Histogram(final_dataset.loc[final_dataset.AgeGroup=='20-29'],x='sum_score'), row=1, col=1)
fig.update_layout(height=600, width=600, title_text="Total Score Histograms by Age Subgroup")
st.plotly_chart(fig)








