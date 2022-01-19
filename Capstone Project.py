"""
Nicole Ciaccia 

Data Science Academy Capstone Project

January 2022
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot

# Read in data
df = pd.read_csv(r"C:\Users\NCiaccia\Documents\Data Science Academy\Week 8-10\data_capstone_dsa2021_2022.csv")

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




# Get dataframe of just the scores
final_dataset2=final_dataset.iloc[:, 20:] 
final_dataset2.drop(columns=['state2', 'gender', 'home_computer', 'state', 'age', 'state_final'], inplace=True) 

# Run Pearson correlation
corrs=final_dataset2.corr(method ='pearson')
# mask=np.zeros_like(corrs)
# mask[np.triu_indices_from(mask)] = True


# Only keep the last row - correlation between item and total score
corrs2=corrs.loc[['sum_score']]
#get rid of sum_score column since it is unneeded
corrs2.drop(columns=['sum_score'], inplace=True)

#get rid of prefix in col names and transpose for easier graphing
corrs2.columns = corrs2.columns.str.strip('gs_')
corrs3 = corrs2.T
corrs3.reset_index(inplace=True)
corrs3.rename(columns={'index':'Item'}, inplace=True)

#Simple bar char of correlations for each item
fig, ax = plt.subplots()
ax.bar('Item','sum_score', data=corrs3)
ax.set(ylim=[0,1], title='Simple Bar Chart - Correlation between Items and Total Score',
       ylabel='Total Score', xlabel='Item')

plt.show()

# Interactive bar chart (need to use plotly offline and plot(fig1) because otherwise cant see plotly charts in spyder)
fig1=px.bar(corrs3,x='Item', y='sum_score', title ="Interactive Bar Chart - Correlation between Items and Total Score")
plot(fig1)



bins_list = [0, 300, 600, 900, 1200, 1500, 3000, 4600]
#bins_list =np.arange(0,4800,300)
plt.hist(final_dataset.rt_total,edgecolor="black", color="m", bins=bins_list)
plt.xticks(bins_list, rotation = 90)
plt.title("Histogram of Total Testing Time")
plt.xlabel("Total Testing Time")
plt.ylabel("Frequencies")




# Interactive Bar chart of Mean Scores by State and Gender
state_gender_means = final_dataset.groupby(['gender', 'state_final'])['sum_score'].mean().reset_index()
f_means = state_gender_means.loc[state_gender_means['gender'] == 'Female']
m_means = state_gender_means.loc[state_gender_means['gender'] == 'Male']

state_gender_means2=f_means.append(m_means)


fig2=px.bar(state_gender_means2,x='gender',y='sum_score',color='gender',animation_frame='state_final',title ="Mean Scores by Gender and State")
plot(fig2)

testing = final_dataset['rt_total'].value_counts()


datax = final_dataset['gs_1'].value_counts()
datay = pd.DataFrame({'gs_1': datax.index,'Frequency': datax.values,'Percent': ((datax.values/datax.values.sum())*100).round(2)})
mn=final_dataset.groupby(by="gs_1",as_index=False).mean('sum_score')
mn = mn.rename(columns={'sum_score': 'sum_score_Mean'})
mn=mn[['gs_1','sum_score_Mean']]
std=final_dataset.groupby(["gs_1"])['sum_score'].std()
std2=pd.DataFrame(std)
std2 = std2.rename(columns={'sum_score': 'sum_score_STD'})
mn_std=mn.merge(std2,left_index=True, right_index=True)
freq_mn_std=datay.merge(mn_std,left_on='gs_1', right_on='gs_1')
freq_mn_std['Item'] = "gs_1"
freq_mn_std = freq_mn_std.rename(columns={'gs_1': 'Score'})
freq_mn_std.sort_values(by=['Score'], inplace=True)






for i in range(1,21):
    datax = final_dataset['gs_%s'%i].value_counts()
    datay = pd.DataFrame({'gs_%s'%i: datax.index,'Frequency': datax.values,'Percent': ((datax.values/datax.values.sum())*100).round(2)})
    mn=final_dataset.groupby(by="gs_%s"%i,as_index=False).mean('sum_score')
    mn = mn.rename(columns={'sum_score': 'sum_score_Mean'})
    mn=mn[['gs_%s'%i,'sum_score_Mean']]
    std=final_dataset.groupby(["gs_%s"%i])['sum_score'].std()
    std2=pd.DataFrame(std)
    std2 = std2.rename(columns={'sum_score': 'sum_score_STD'})
    mn_std=mn.merge(std2,left_index=True, right_index=True)
    freq_mn_std=datay.merge(mn_std,left_on='gs_%s'%i, right_on='gs_%s'%i)
    freq_mn_std['Item'] = "gs_%s"%i
    freq_mn_std = freq_mn_std.rename(columns={'gs_%s'%i: 'Score'})
    freq_mn_std.sort_values(by=['Score'], inplace=True)
    if i == 1:
        all_freq_mn_std=freq_mn_std
    else:
        all_freq_mn_std=all_freq_mn_std.append(freq_mn_std)

fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=all_freq_mn_std.values, colLabels=freq_mn_std.columns, loc='center')
fig.tight_layout()
plt.show()




