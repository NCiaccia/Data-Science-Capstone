"""
Nicole Ciaccia 

Data Science Academy Capstone Project

January 2022
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Read in data
df = pd.read_csv("data_capstone_dsa2021_2022.csv")

# Cleaning up state column. First uppercase all for easier standardization
df['state'] = df['state'].str.upper()

# Create a dictionary  of all state names and abbreviations
us_state_to_abbrev = {
	    "ALABAMA": "AL",  "ALASKA": "AK",  "ARIZONA": "AZ",  "ARKANSAS": "AR",
	    "CALIFORNIA": "CA", "COLORADO": "CO",  "CONNECTICUT": "CT", "DELAWARE": "DE",
	    "FLORIDA": "FL",  "GEORGIA": "GA",  "HAWAII": "HI",  "IDAHO": "ID",  "ILLINOIS": "IL",
	    "INDIANA": "IN",  "IOWA": "IA",  "KANSAS": "KS",  "KENTUCKY": "KY",  "LOUISIANA": "LA",
	    "MAINE": "ME",  "MARYLAND": "MD",  "MASSACHUSETTS": "MA",  "MICHIGAN": "MI",  "MINNESOTA": "MN",
	    "MISSISSIPPI": "MS",  "MISSOURI": "MO",  "MONTANA": "MT",  "NEBRASKA": "NE",  "NEVADA": "NV",
	    "NEW HAMPSHIRE": "NH",  "NEW JERSEY": "NJ",  "NEW MEXICO": "NM",  "NEW YORK": "NY",  "NORTH CAROLINA": "NC",
	    "NORTH DAKOTA": "ND",  "OHIO": "OH",  "OKLAHOMA": "OK",  "OREGON": "OR",  "PENNSYLVANIA": "PA",
	    "RHODE ISLAND": "RI",  "SOUTH CAROLINA": "SC",  "SOUTH DAKOTA": "SD",  "TENNESSEE": "TN",  "TEXAS": "TX",
	    "UTAH": "UT",  "VERMONT": "VT",  "VIRGINIA": "VA",  "WASHINGTON": "WA",  "WEST VIRGINIA": "WV",
	    "WISCONSIN": "WI",  "WYOMING": "WY",  "PUERTO RICO": "PR",
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

final_dataset['state_final'] = np.where(final_dataset['state2'].isin(abbrevs), final_dataset['state2'], "_Unknown_State")

# (end up with 92 observations with no state2, but that is the best we can do)
check_state = final_dataset.loc[final_dataset['state_final'] == "_Unknown_State"]

# Get dataframe of just the scores and just timings (used in graphs later on)
scores=final_dataset.iloc[:, 20:] 
scores.drop(columns=['state2', 'gender', 'home_computer', 'state', 'age', 'state_final'], inplace=True) 

timings=final_dataset.iloc[:, :19] 

# Create AgeGroup variable
bins= [0,20,30,40,50,60,110]
labels = ['Under 20','20-29','30-39','40-49', '50-59','60+']
final_dataset['AgeGroup'] = pd.cut(final_dataset['age'], bins=bins, labels=labels, right=False)

################################ DATA CLEAN UP FINISHED ################################
########################################################################################
################################ BEGIN USING STREAMLIT #################################

st.set_page_config(layout='wide')


# Title for dashboard
st.title("Welcome to my Data Science Dashboard!")
st.markdown('### This dashboard examines several questions about the provided data to see \
             if there are any apparent relationships between variables.')
st.markdown('By: Nicole Ciaccia - PAR PEP team - January 2022')


st.text("")
st.text("")
st.markdown('##### Investigation 1: Do any individual items have a high correlation with the total score?')

# Run Pearson correlation
corrs=scores.corr(method ='pearson')

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
fig1=px.bar(corrs3,x='Item', y='correlation', color='correlation',  color_continuous_scale=["blue", "white", "red"],  
            title ="Correlations Between Each Item and Total Score")
# fig1=px.bar(corrs3,x='Item', y='correlation', color='correlation',  range_color=[0,1], color_continuous_scale=["blue", "white", "red"],  
#             title ="Correlations Between Each Item and Total Score")
fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20))

st.text("")
a_col, b_col = st.columns(2)
with a_col:
    st.plotly_chart(fig1)
with b_col:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown("**Findings:** For Praxis tests, we consider anything with a correlation above 0.25 or 0.3 to\
            be a relatively good item. Using that same criteria on this data, it looks like most items\
            have a decent correlation with the total score. Items 4 and 7 stand out as the highest; Items\
            5, 15 and 16 are on the cusp, but still relatively good. The only item that stands out is \
            Item 20 which clearly has the worst correlation. This could in part be due to its postition at\
            the end of the test. Let's keep investigating item stats to see what we uncover.")

st.text("")

# Get AIS for each item
for i in range(1,21):
    datax = final_dataset['gs_%s'%i].value_counts()
    datay = pd.DataFrame({'gs_%s'%i: datax.index,'N': datax.values,'AIS': ((datax.values/datax.values.sum())*100).round(2)})
    datay['Item'] = "%s"%i
    datay= datay.loc[datay['gs_%s'%i] == 1]
    datay= datay[["Item", "AIS", "N"]] 
    if i == 1:
       ais=datay
    else:
        ais=ais.append(datay)

# Get Avg time for each item
temp1 = timings.mean(axis=0)
temp1.reset_index(drop=True, inplace=True) 
temp2 = pd.Series(range(2,21))
frame = {'Item': temp2, 'Avg_Time': temp1}
time_means = pd.DataFrame(frame)

st.markdown('##### Investigation 2: How do the average item scores and item timings compare for each item?')
trace1 = go.Bar(
    x=ais['Item'],
    y=ais['AIS'],
    name='AIS (%)',
    marker=dict(color='paleturquoise')
)
trace2 = go.Scatter(
    x=time_means['Item'],
    y=time_means['Avg_Time'],
    name='Avg Time (seconds)',
    marker=dict(color='tomato'),
    yaxis='y2'
)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(trace1)
fig2.add_trace(trace2,secondary_y=True)
fig2['layout'].update(height = 550, width = 1000, xaxis_title="Item", yaxis_title="Average Item Score (AIS)",
                      title_text='Average Item Score and Time Spent on Each Time', title_x=0.25, 
                      margin=dict(l=20, r=20, t=30, b=20))
fig2.update_yaxes(title_text="Average Time Spent on Item", secondary_y=True)

st.text("")
col1, col2 = st.columns([2.5,7])
with col2:
    st.plotly_chart(fig2)


    
st.markdown("**Findings:** Looking at the Average Item Score and Timing plot, we can see a lot\
            more information about Items 20 and 15. Almost 100% of candidates got them correct and \
            they were the 2 items the candidates spent the least amount of time on. From these visualizations\
            we can conclude that these were 2 very easy items, so it makes sense that they were marked as \
            non discriminating items in the graph above. Item 16 also had a very high AIS and low average time \
            so it is a reasonable assumption that this item was easy as well. The correlations were more about \
            item difficulty as opposed to item position. A new fact about the items that this plot reveals is that \
            Item 13 was by far the most difficult item and one that candidates spent a fair amout of time on.")
st.markdown("Note: No Item Timings were provided for the first item")
st.text("")
st.text("")


# Examine impact of Age on Score and Testing Time
st.markdown('##### Investigation 3: Does age impact total score?')
fig3= px.histogram(final_dataset,x='sum_score',color='AgeGroup',facet_col='AgeGroup', facet_col_wrap=3, height=700, width=1200, 
             category_orders={"AgeGroup": labels}, title="Total Score Histograms by Age Subgroup")

c_col, d_col = st.columns((2.4,1))
with c_col:
    st.plotly_chart(fig3)

with d_col:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown("**Findings:** Distributions for each age category seem to be relatively simlilar\
            in that the data is very left skewed for each histogram. It appears that whatever \
            this exam was, the questions were not very difficult. No one particular age range seems to have a \
            distribution that shows they are especially more or less proficient than the others. Though \
            the N for Under 20 is very low, it is interesting that no one in this age category got \
            below a 16. However the  \n 50-59 and 60+ groups also have very few candidates below 16 so \
            it's not as if being younger seems to be an extreme advantage.  \n \n You can also see from \
            these plots that the majority of the population falls within the middle age groups which \
            could be interesting depending on what exam the data is actually for.")

st.text("")
st.markdown('##### Investigation 4: Did any states outperform the others?')
state_means = final_dataset.groupby('state_final').agg({'sum_score': ['mean', 'min', 'max', 'count']})
state_means.columns = state_means.columns.droplevel(0)
state_means.reset_index(inplace=True)
state_means = state_means[state_means.state_final != '_Unknown_State']


fig4a = px.choropleth(state_means, color='mean', locations='state_final', locationmode = 'USA-states',
                    scope='usa', title="Average Total Score by US State", hover_data=['state_final', 'mean', 'count'],
                    labels={"state_final": "State", "mean": "Average Total Score", "count": "N"},
                    color_continuous_scale= 'sunset')
fig4a.update_layout(dragmode = False, margin=dict(l=20, r=20, t=30, b=20), title_x=0.5, title_y=.95)


fig4b = px.choropleth(state_means, color='count', locations='state_final', locationmode = 'USA-states',
                    scope='usa', title="Counts by US State", hover_data=['state_final', 'mean', 'count'],
                    labels={"state_final": "State", "mean": "Average Total Score", "count": "N"},
                    color_continuous_scale= 'sunset')
fig4b.update_layout(dragmode = False, margin=dict(l=20, r=20, t=30, b=20), title_x=0.5, title_y=.95)

a_col, b_col = st.columns(2)
with a_col:
    st.plotly_chart(fig4a)
with b_col:
    st.plotly_chart(fig4b)


fig4c=px.bar(state_means,x='state_final', y='mean', height=600, width=1500, color='count', 
            hover_data=["mean", "min", "max", "count"], title ="Total Score Mean by State with Counts",
            labels={"state_final": "State", "mean": "Average Total Score", "max": "Max", "min": "Min", "count": "N"})
fig4c.update_layout(margin=dict(l=20, r=20, t=30, b=20), title_x=0.5)

col1, col2 = st.columns([1,12])
with col2:
    st.plotly_chart(fig4c)
            

st.markdown("**Findings:** No state stands out particularly as an outlier. The one visual exception would be \
            Puerto Rico, but then when you hover over that bar, you can see the N=1 so that is a very small \
            sample size. Utah and Montana have the highest means, but only slightly higher than other states \
            and again N is small so no grand conclusions can be drawn. Based on this plot, if analysis was \
            being done to see if the test was biased by geographical region in any way, \
            it would be reassuring that there doesnt seem to be any evidence that results are skewed based on \
            location. This graph does give some more information about the testing population, indicating that\
            there are a few states with much higher representation than others, namely California, Florida, \
            New York and Texas.")


st.text("")
st.markdown('##### Investigation 5: Does gender have any impact on the data?')
st.markdown("- First let's look at the distribution of Males and Females in the data:")

gender_means = final_dataset.groupby('gender').agg({'sum_score': ['mean', 'min', 'max', 'count', 'std']})
gender_means.columns = gender_means.columns.droplevel(0)
gender_means.reset_index(inplace=True)
gender_means['AgeGroup']='Total'
age_gender_means = final_dataset.groupby(['gender', 'AgeGroup']).agg({'sum_score': ['mean', 'min', 'max', 'count',  'std']})
age_gender_means.columns = age_gender_means.columns.droplevel(0)
age_gender_means.reset_index(inplace=True)


fig5 = px.bar(age_gender_means, x="AgeGroup", color="gender", y="count", barmode='group', height=600, 
             color_discrete_map={'Male': 'skyblue','Female': 'thistle'},
             labels={"count": "N", "AgeGroup": "Age Group", "gender": "Gender"})
fig5.update_layout(title_text="Male and Female by Age Group", title_x=0.47)

fig6 = px.pie(gender_means, values='count', names='gender',color='gender', labels={"count": "N", "gender": "Gender"}, 
              color_discrete_map={'Male': 'skyblue','Female': 'thistle'})
fig6.update_traces(textposition='inside', textinfo='percent+label')
fig6.update_layout(title_text="Male and Female for Total Group", title_x=0.47)

a_col, b_col = st.columns(2)
with a_col:
    st.plotly_chart(fig5)
with b_col:
    st.plotly_chart(fig6)

st.markdown("**Findings:** Based on these plots it appears the data is about evenly distributed between males \
            and females. There is some disparity in the 20-29 and 50-59 year age ranges, but in most groups \
            it is relatively similar and the overall distribution is about equal.")

st.text("")
st.markdown("Now lets see if gender has any impact on scores:")

age_gender_means2=age_gender_means.append(gender_means)
age_gender_means2.sort_values(by=['AgeGroup'], inplace=True)
age_gender_means2["new"] = range(2,len(age_gender_means2)+2)
age_gender_means2['new'] = np.where(age_gender_means2['AgeGroup']== "Under 20", 0,(age_gender_means2['new']))
age_gender_means2=age_gender_means2.sort_values("new").drop('new', axis=1)
age_gender_means2 = age_gender_means2[["gender", "AgeGroup", "count", "min", "max", "mean", "std"]]
age_gender_means2.columns = map(str.capitalize, age_gender_means2.columns)
age_gender_means2 = age_gender_means2.rename({'Agegroup': 'Age Group', 'Count': 'N', 'Std': 'St. Dev.'}, axis=1)  


# HACK: double the dataframe for getting total in box plot
temp=final_dataset.copy(deep=True)
temp['AgeGroup']='Total'
df_for_box=final_dataset.append(temp)
labels2=labels.copy()
labels2.append('Total')
fig7 = px.box(df_for_box, y="AgeGroup", x="sum_score", color="gender",  category_orders={"AgeGroup": labels2},
              color_discrete_map={'Male': 'skyblue','Female': 'thistle'}, title= "Total Score by Gender", 
              labels={"sum_score": "Total Score", "AgeGroup": "Age Group", "gender": "Gender"})

fig7.update_layout(margin=dict(l=20, r=20, t=30, b=20), title_x=0.5)


a_col, b_col = st.columns(2)
with a_col:
    st.text("")
    st.plotly_chart(fig7)
with b_col:
    #st.dataframe(age_gender_means2, height=400)
    st.markdown('Total Score Statistics by Gender')
    st.dataframe(age_gender_means2.assign(hack='').set_index('hack'), height=450)

st.markdown("**Findings:** It appears that Males did slightly better on this test than Females \
            as the overall mean for Males is slightly higher, and looking at the box plot, the \
            Interquartile Range for each Age Group is slightly higher. When you look at the plot \
            for the Total group, you can see the overall distribution of scores is quite similar \
            though in that the IQR is identical, though the median for Males is 18 as opposed to 17 \
            for Females.")


st.text("")
st.markdown('If there is interest in breakdown by state, Mean and Time data by Gender has also been \
            plotted. Select one state or multiple to see how they compare.')

def state_gender (var):
    means = final_dataset.groupby(['gender', 'state_final'])[var].mean().reset_index()
    f_means = means.loc[means['gender'] == 'Female']
    m_means = means.loc[means['gender'] == 'Male']
    means2=f_means.append(m_means)
    means2.sort_values(by=['state_final'], inplace=True)
    means2 = means2[means2.state_final != '_Unknown_State']
    means2.sort_values(by=['state_final', 'gender'], inplace=True)
    return means2

state_gender_means= state_gender('sum_score')
state_gender_times= state_gender('rt_total')


states=st.multiselect('Select 1 or more States:', abbrevs, default='AL') #create selector box
state_gender_means2=state_gender_means.query('state_final==@states') #query the data using the selector box
state_gender_times2=state_gender_times.query('state_final==@states') 

fig8=px.bar(state_gender_means2,x='state_final',y='sum_score', color='gender', barmode='group', title ="Mean Scores by Gender and State",
            color_discrete_map={'Male': 'skyblue','Female': 'thistle'}, 
            labels={"sum_score": "Total Score", "state_final": "State", "gender": "Gender"})
fig8.update_layout(margin=dict(l=20, r=20, t=30, b=20), title_x=0.5)
fig9=px.bar(state_gender_times2,x='state_final',y='rt_total', color='gender', barmode='group', title ="Average Time by Gender and State",
            color_discrete_map={'Male': 'skyblue','Female': 'thistle'}, 
            labels={"rt_total": "Average Time", "state_final": "State", "gender": "Gender"})
fig9.update_layout(margin=dict(l=20, r=20, t=30, b=20), title_x=0.5)

a_col, b_col = st.columns(2)
with a_col:
    st.text("")
    st.plotly_chart(fig8)
with b_col:
    st.text("")
    st.plotly_chart(fig9)


st.text("")
st.text("")
st.text("")
st.markdown('##### Investigation 6: What are the relationships between Time with Age and Performance?')

def time_age_score(var):
    df = final_dataset.groupby([var]).agg({'rt_total': ['count', 'mean']})
    df.columns = df.columns.droplevel(0)    
    df.reset_index(inplace=True)
    df.rename(columns={'count':'N', 'mean':'Avg_Time'}, inplace=True)
    return df

time_age=time_age_score('age')
time_age=time_age[time_age.age != 0]

time_age2=time_age_score('AgeGroup')

fig10 = px.scatter(time_age, x="age", y="Avg_Time", size='N', labels={"Avg_Time": "Average Time", "age": "Age"})
fig10.update_layout(margin=dict(l=20, r=20, t=30, b=20), title='Average Time by Age', title_x=0.5)

fig10a = px.scatter(time_age2, x="AgeGroup", y="Avg_Time", size='N', labels={"Avg_Time": "Average Time", "AgeGroup": "Age Group"})
fig10a.update_layout(margin=dict(l=20, r=20, t=30, b=20), title='Average Time by Age Group', title_x=0.5)

a_col, b_col = st.columns(2)
with a_col:
    st.text("")
    st.plotly_chart(fig10)
with b_col:
    st.text("")
    st.plotly_chart(fig10a)
    
st.markdown("**Findings:** Looking at the Time vs Age plot, there does not really seem to be any stand out connections. \
            The average time for each age is in the same realm with a few individual ages being higher in the older ages \
            but overall there does not appear to be a clear relationship. However, when Time is plotted vs *Age Group*, \
            a connection becomes clear. You can see very distinctly that the average time spent testing increases \
            as the Age Group gets older. This is good evidence that sometimes you must look at the same variables \
            from different angles to see if any connections are present.")

time_score=time_age_score('sum_score')

fig11 = px.scatter(time_score, x="sum_score", y="Avg_Time", size='N', labels={"Avg_Time": "Average Time", "sum_score": "Total Score"})
fig11.update_layout(margin=dict(l=20, r=20, t=30, b=20), title='Average Time by Score', title_x=0.5)
fig11.update_traces(marker_color='green', opacity=0.5)

a_col, b_col = st.columns(2)
with a_col:
    st.text("")
    st.plotly_chart(fig11)
with b_col:
    st.text("")
    st.text("")    
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown("**Findings:** In the Time vs Total Score plot, there also appears to be some conclusions to draw. \
            It seems that scores continue to rise as time spent increases, to a point. A limit is reached and \
            then you can see as perfect or nearly perfect scores are approached, the average time spent actually \
            drops. That is, the most proficient test takers were able to complete the exam in less time than the \
            group right below them.")


st.text("")
st.text("")
st.markdown('##### Investigation 7: Can we use any of the variables to predict performance?')
st.markdown("Looking at the investigations above, item 7 had the best correlation with total score and there seemed \
            to be some connections between Age Group and total time spent testing. Let's see how good of a \
            predictor these 3 variables are in predicting a test taker's total score.")
            
from sklearn import preprocessing
import warnings; warnings.simplefilter('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # for Random Forest
from sklearn.ensemble import GradientBoostingClassifier # for Gradient Boosting Machine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

f = final_dataset[['gs_7', 'rt_total', 'AgeGroup']]
t = final_dataset[['sum_score']]

a={'Under 20': 0,'20-29': 1,'30-39': 2,'40-49': 3, '50-59':4,'60+':5}
f['age_converted'] = f['AgeGroup'].map(a)
f.drop(columns=['AgeGroup'], inplace=True)


f = preprocessing.scale(f) #normalize to 0 mean and unit standard deviation

model_SVM = SVC()
model_RF = RandomForestClassifier()
model_GBM = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(f, t, test_size=0.5, random_state=42)

tmp1 = model_SVM.fit(X_train,y_train)
tmp2 = model_RF.fit(X_train,y_train)
tmp3 = model_GBM.fit(X_train,y_train)

y_pred_SVM = model_SVM.predict(X_test)
y_pred_RF = model_RF.predict(X_test)
y_pred_GBM = model_GBM.predict(X_test)

# --- Model evaluation and selection ---
model_evals = pd.DataFrame(columns = ['Model', 'Accuracy', 'Kappa'])
model_evals['Model']=['SVM', 'RF', 'GBM' ]

acc_svm = accuracy_score(y_pred_SVM, y_test)
acc_rf = accuracy_score(y_pred_RF, y_test)
acc_gbm = accuracy_score(y_pred_GBM, y_test)

kap_svm = cohen_kappa_score(y_pred_SVM, y_test)
kap_rf = cohen_kappa_score(y_pred_RF, y_test)
kap_gbm = cohen_kappa_score(y_pred_GBM, y_test)
model_evals['Accuracy']=[acc_svm, acc_rf, acc_gbm]
model_evals['Kappa']=[kap_svm, kap_rf, kap_gbm]


model_predictions=pd.DataFrame({'SVM': y_pred_SVM, 'RF': y_pred_RF, 'GBM': y_pred_GBM})
y_test.reset_index(drop=True, inplace=True)
act_pred = pd.merge(y_test, model_predictions, left_index=True, right_index=True)


st.markdown("**Method:** Three Supervised Learning Models were used- Support Vector Machine (SVC), Random Forrest (RF) \
            and Gradient Boosting Method (GBM). The predicted values for Total Score are plotted below against the actual \
            total scores for each of the models.")
st.text("")

fig12=px.scatter(act_pred,x='sum_score', y='SVM', title ="Support Vector Machine Model", height=350, width=475, labels={"sum_score":"Observed", "SVM":"Predicted"})
fig12.update_layout(margin=dict(l=10, r=10, t=30, b=20), title_x=0.5)
fig12.update_traces(marker_color='lightcoral', opacity=0.5)


fig13=px.scatter(act_pred,x='sum_score', y='RF', title ="Random Forrest Model", height=350, width=475, labels={"sum_score":"Observed", "RF":"Predicted"})
fig13.update_layout(margin=dict(l=10, r=10, t=30, b=20), title_x=0.5)
fig13.update_traces(marker_color='teal', opacity=0.5)

fig14=px.scatter(act_pred,x='sum_score', y='GBM', title ="Gradient Boosting Machine Model", height=350, width=475, labels={"sum_score":"Observed", "GBM":"Predicted"})
fig14.update_layout(margin=dict(l=10, r=10, t=30, b=20), title_x=0.5)
fig14.update_traces(marker_color='dodgerblue', opacity=0.5)

#below code is to hide row indices
#CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

a_col, b_col, c_col, d_col = st.columns([4,4,4,2.5])
with a_col:
    st.plotly_chart(fig12)
with b_col:
    st.plotly_chart(fig13)
with c_col:
    st.plotly_chart(fig14)
with d_col:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(model_evals.style.set_precision(2))

st.markdown("**Findings:** As you can see based on the fact that the scatter plots are all over the place with \
            no particular visible line or pattern, these models did a terrible job of predicting total score.\
            Looking at the Accuracy and Kappa for each model, they are horribly low.  \n While I have been able to \
            find some relationships between variables in this dataset, it looks like a lot more work would need to \
            be done either adjusting the parameters of the models or finding better features in order to make more \
            accurate predictions.")







# trace1 = go.Scatter(
#     x=y_test['obs'],
#     y=y_test['sum_score'],
#     mode='markers',
#     marker=dict(color='tomato'),
# )
# trace2 = go.Scatter(
#     x=df_svm['obs'],
#     y=df_svm['sum_score'],
#     mode='markers',
#     marker=dict(color='green'),
# )

# fig12 = make_subplots()
# fig12.add_trace(trace1)
# fig12.add_trace(trace2)
# # fig2['layout'].update(height = 550, width = 1000, xaxis_title="Item", yaxis_title="Average Item Score (AIS)",
# #                       title_text='Average Item Score and Time Spent on Each Time', title_x=0.25, 
# #                       margin=dict(l=20, r=20, t=30, b=20))
# # fig2.update_yaxes(title_text="Average Time Spent on Item", secondary_y=True)
# st.plotly_chart(fig12)


