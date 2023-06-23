# Import necessary libraries
import pandas as pd 
import trueskill
from trueskill import Rating, rate,setup
import datetime
import warnings
import json
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st 
import itertools
import logging # Import the logging library
import plotly.graph_objects as go
import ast
from dateutil.relativedelta import relativedelta
from pandas.api.types import CategoricalDtype
from datetime import date

#import streamlit.components.v1 as components
st.set_page_config(page_title="Data Explorer", layout="wide",initial_sidebar_state="collapsed")
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',unsafe_allow_html=True)


import logging

# Make sure the directory exists. If it doesn't, create it.
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure the logging module.
logging.basicConfig(
    filename='logs/app.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


st.markdown(f"""
<div style='text-align: center'>
<div class="card text-white bg-danger mb-3" >
  <div class="card-header"> <h2> True-skill </h2></div>    
    <p class="card-text"></p>
  </div>
  </div>
""", unsafe_allow_html=True)




logger = logging.getLogger(__name__)


# Suppress warnings
import warnings
warnings.filterwarnings('ignore', module='numpy')
warnings.filterwarnings('ignore')


logger.info('Starting to read CSV files') # Add a logging statement
# Read the CSV files
overall_submissions=pd.read_csv('All Submissions-main.csv')
bounty_details=pd.read_csv('Bounty-main.csv')

logger.info('Merging CSV files') # Add a logging statement
# Merge the CSV files based on the 'Bounty' column
overall_submissions=overall_submissions.merge(bounty_details,left_on='Bounty',right_on='Name')

logger.info('Converting date format and sorting the dataframe') # Add a logging statement
# Convert the 'End Date' column to datetime format and sort the dataframe by it
overall_submissions['End Date']=pd.to_datetime(overall_submissions['End Date'])
overall_submissions['End Date'] = overall_submissions['End Date'].dt.strftime("%d-%m-%Y %H:%M:%S")
overall_submissions['End Date']=pd.to_datetime(overall_submissions['End Date'])
overall_submissions=overall_submissions.sort_values(by='End Date',ascending=True)

logger.info('Converting wallet columns to lowercase') # Add a logging statement
# Convert 'MetaMask Wallet' and 'xMetric Wallet' columns to lowercase
overall_submissions['MetaMask Wallet'] = overall_submissions['MetaMask Wallet'].str.lower()
overall_submissions['xMetric Wallet'] = overall_submissions['xMetric Wallet'].str.lower()

logger.info('Filtering out specific rows from overall_submissions')  # Add a logging statement
# Filter out rows where 'Grading Observations v8' and 'Grading Observation v7' are NaN and 'Bounty' is not 'NEAR - 14. Wen Hunt? Now. '
# we omit near- wen hunt cause it had -1 as score for everyone intentionally
overall_submissions=overall_submissions[overall_submissions['Grading Observations v8'].isna()]
overall_submissions=overall_submissions[overall_submissions['Grading Observation v7'].isna()]
overall_submissions=overall_submissions[overall_submissions['Bounty']!='NEAR - 14. Wen Hunt? Now. ']
overall_submissions_calc=overall_submissions.copy()


logger.info('Creating a new dataframe for discord handles and sorting by Created At')  # Add a logging statement
# Create a new dataframe for discord handles and sort by 'Created At'
discord_handles=overall_submissions[['Discord Handle','xMetric Wallet','Created At']]
discord_handles['Created At'] = pd.to_datetime(discord_handles['Created At'], format="%d/%m/%Y %H:%M")
discord_handles['Created At'] = discord_handles['Created At'].dt.strftime("%d-%m-%Y %H:%M:%S")
discord_handles=discord_handles.sort_values(by='Created At',ascending=True)
discord_handles.dropna(inplace=True)
discord_handles.drop_duplicates(subset='xMetric Wallet',keep='last',inplace=True)



logger.info('Fetching data from Flipside Crypto API')  # Add a logging statement
# Fetch data from Flipside Crypto API
on_chain_reviews=pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/37106c4d-afc1-49db-9412-b32a7b0a9ab8/data/latest')
on_chain_bounty=pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/a40ccefa-b556-4dd0-b0a6-f38b494309b2/data/latest')
on_chain_submission=pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/cd9455dd-7363-470a-8399-dcb7c5474842/data/latest')

logger.info('Merging data from API responses')  # Add a logging statement
# Merge data from API responses
review_and_bounty=on_chain_reviews.merge(on_chain_bounty,left_on='BOUNTY_IDD',right_on='Searchable_id',how='left')
submission_details=review_and_bounty.merge(on_chain_submission,on=['BOUNTY_ID','SUBMISSION_ID','ANALYST'],how='left')
submissions=submission_details.merge(discord_handles,left_on='ANALYST',right_on='xMetric Wallet',how='left')

logger.info('Filling missing values in Discord Handle with ANALYST')  # Add a logging statement
# Fill missing values in 'Discord Handle' with 'ANALYST'
submissions['xMetric Wallet']=submissions['ANALYST']
submissions['Discord Handle'].fillna(submissions['ANALYST'],inplace=True)
submissions['Chain/Project']=submissions['Chain/Project'].apply(lambda x:str(x))

# Group the submissions dataframe
logger.info('Grouping the submissions dataframe')  # Add a logging statement
submissions=submissions[['SUBMISSION_URL','Challenge URL','Challenge','ANALYST','Discord Handle','Review Deadline','Chain/Project','SCORE']]
submissions=submissions.groupby(by=['SUBMISSION_URL','Challenge URL','Challenge','ANALYST','Discord Handle','Review Deadline','Chain/Project'],as_index=False).mean()

# Create a subset of overall_submissions_calc
logger.info('Creating a subset of overall_submissions_calc')  # Add a logging statement
overall_submissions_calc=overall_submissions_calc[['Bounty',  'Public Result(s)', 'Overall Avg. Grade', 'Discord Handle','End Date','Bounty Program Name','xMetric Wallet']]
#Setting '-' as the challenge url for the challenges launched in notion and later setting it to the same name as bonuty.
overall_submissions_calc['Challenge_url']='-'
overall_submissions_calc.loc[overall_submissions_calc['Challenge_url'] == '-', 'Challenge_url'] = overall_submissions_calc['Bounty']

# Create a subset of submissions and rename its columns
logger.info('Creating a subset of submissions and renaming its columns')  # Add a logging statement
submissions=submissions[['Challenge','SUBMISSION_URL', 'SCORE', 'Discord Handle','Review Deadline','Chain/Project','ANALYST','Challenge URL']]
submissions.columns=['Bounty',  'Public Result(s)', 'Overall Avg. Grade', 'Discord Handle','End Date','Bounty Program Name','xMetric Wallet','Challenge_url']

# Concatenate the dataframes and filter for certain 'Bounty Program Name' values
logger.info('Concatenating the dataframes and filtering for certain Bounty Program Name values')  # Add a logging statement
combined=pd.concat([overall_submissions_calc,submissions],axis=0,join='outer')


processed_data=combined.copy()

 # Add a logging statement




# Function to convert string-represented lists into actual lists
def parse_lists(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [val]
# Apply the function to the 'Bounty Program Name' column
processed_data['Bounty Program Name'] = processed_data['Bounty Program Name'].apply(parse_lists)
# Flatten the lists in 'Bounty Program Name' column to get a list of all program names
all_program_names = list(itertools.chain(*processed_data['Bounty Program Name'].tolist()))
# Get unique program names
unique_program_names = list(set(all_program_names))
unique_program_names.append('All')




c1,c2=st.columns(2)
default_value = [unique_program_names[-1]]
chain = c1.multiselect('Project', unique_program_names, default=default_value)
include_notion=c1.checkbox('Include notion',value=True)

number_of_months=c2.select_slider("Number of Previous Months", options=[0,1,3,6,12,24],value=3)
all_time=c2.checkbox('All time',value=True)
last_date=date.today()
if all_time==True:
    start_date=datetime.date(2022, 1, 1)
elif number_of_months > 0:
    start_date= last_date-relativedelta(months=number_of_months)
else:
    st.write('enter date correctly')

# c2.write('You selected {} Months'.format(number_of_months))

analyst_profile,leader_board,contact,credits=st.tabs(['Analyst Profile','LeaderBoard','Contact','Credits'])
with analyst_profile: 

    c1,c2=st.columns(2)
    name=c1.selectbox('Discord Handle',processed_data['Discord Handle'].unique())
    
    address=c2.selectbox('Wallet Address',processed_data['xMetric Wallet'].unique())

    st.markdown("---")




    logger.info('Filtering the dataset to include chain/project')  # Add a logging statement
    if chain[0]=='All':
        Bounty_Program_Name=combined['Bounty Program Name'].unique()

    else:
        try:
            chain.remove('All')
        except:
            pass
        Bounty_Program_Name=chain
    pattern = '|'.join(Bounty_Program_Name)
    processed_data = combined[combined['Bounty Program Name'].str.contains(pattern, na=False, case=False)]


    logger.info('Exclude challenges with less than 2 participants')  # Add a logging statement
    # Get a list of challenge URLs where 'Discord Handle' count is less than 2
    challenge_urls_to_skip = processed_data.groupby(by='Challenge_url').filter(lambda x: x['Discord Handle'].nunique() < 2)['Challenge_url'].unique().tolist()

    # Filter the DataFrame to exclude the challenge URLs in the list
    processed_data = processed_data[~processed_data['Challenge_url'].isin(challenge_urls_to_skip)]
    logger.info('Excluded challenges with less than 2 participants,%s',str(challenge_urls_to_skip)) 


    logger.info('Using time bounds') 
    
    # Convert 'End Date' column to datetime
    processed_data['End Date'] = pd.to_datetime(processed_data['End Date'])

    # Convert start_date to pandas.Timestamp
    start_date = pd.Timestamp(start_date)
    last_date = pd.Timestamp(last_date)

    # Then you can make the comparison
    processed_data=processed_data[(processed_data['End Date']>=start_date) & (processed_data['End Date']<=last_date)]









    # Initialize a dataframe for ELO score and save it to a CSV file
    logger.info('Initializing a dataframe for ELO score and saving it to a CSV file')  # Add a logging statement
    last_elo_score = pd.DataFrame(columns=['Discord Handle', 'mean', 'variance'])
    last_elo_score.to_csv('latest_elo_score.csv')

    


    # Calculate ELO scores for each unique challenge url
    logger.info('Calculating ELO scores for each unique challenge url')  # Add a logging statement

    # Initialize a list for final output 
    data_with_rank=[]
    st.dataframe(processed_data[processed_data['Discord Handle']=='sandesh#8645'].sort_values(by='End Date',ascending=True))

    
    for bounty in processed_data['Challenge_url'].unique():
            # Remainder of the code includes calculations for ELO score
            # Filter for unique challenge url
        logger.debug('Processing challenge: %s', bounty)  # Add a debugging statement
        record_details=[]
        
        challenge_df=processed_data[processed_data['Challenge_url']==bounty]
    #     temp=temp[temp['Bounty Program Name']==Bounty_Program_Name]
        challenge_df=challenge_df.sort_values(by='Overall Avg. Grade',ascending=False)
        challenge_df['Rank']=challenge_df['Overall Avg. Grade'].rank(ascending=False,method='min')
        challenge_df=challenge_df[['Bounty',  'Public Result(s)', 'Overall Avg. Grade', 'Discord Handle','Rank','End Date','Bounty Program Name','xMetric Wallet','Challenge_url']]
        End_Date=challenge_df['End Date'].iloc[0]
        latest_elo_score=pd.read_csv('latest_elo_score.csv')
        elo_score={}
        latest_elo_score=latest_elo_score[['Discord Handle','mean','variance']]


        # Checking if challenge_url is same as bounty name, if it is, then it means it's a challenge launched on notion
        if challenge_df['Challenge_url'].iloc[0]==challenge_df['Bounty'].iloc[0]:
            if include_notion==False:
                continue

            # continue
            
            
            challenge_df=challenge_df.drop_duplicates(subset=['Discord Handle'],keep='first')


            # Iterate over each 'Discord Handle' in the challenge data.
            for handle in (challenge_df['Discord Handle']):

                # If the handle is present in the latest ELO score dataframe, then set the
                # ELO score of the handle as the 'mean' ELO score from the latest ELO score dataframe.
                if handle in latest_elo_score['Discord Handle'].values:
                    elo_score[handle]=Rating(latest_elo_score[latest_elo_score['Discord Handle']==handle]['mean'].values[0]) 

                # If the handle is not present in the latest ELO score dataframe, then set the
                # ELO score of the handle as 100. This can be considered as the default or starting ELO score.
                else:
                    elo_score[handle]=Rating(100)

            # Create an empty list to hold tuples of Rating objects. This is because the 'rate' function 
            # from the TrueSkill package expects a sequence of Rating objects.
            rating_tuple = []

            # Iterate over each ELO score in the elo_score dictionary.
            for elo_object_of_handle in elo_score:
                # Create a new Rating object for the ELO score and append it as a single-item tuple to the 'rating_tuple' list.
                # setup(backend='mpmath')
                rating = Rating(elo_score[elo_object_of_handle])
                rating_tuple.append((rating,))

            # Try to calculate new ELO ratings using the 'rate' function from the TrueSkill package. 
            # The 'rate' function takes in a sequence of Rating objects and a sequence of ranks as arguments.
            try:
                new_ratings = rate(rating_tuple, ranks=challenge_df.Rank)
                
            # If there is an error while calculating the new ELO ratings, catch the exception and print out the details.
            except Exception as e:
                logger.error('Error calculating new ratings for challenge: %s. Error: %s', bounty, e)  # Add an error statement
                continue

            # Set a counter to 0. This counter is used to index the new ratings.
            j=0

            # Iterate over each key (which is a 'Discord Handle') in the 'elo_score' dictionary.
            for i in elo_score.keys():
                # Create an empty list to hold the details of each record.
                record_details=[]

                try:
                    # If the 'Discord Handle' is not present in the 'latest_elo_score' dataframe,
                    # add a new row to the dataframe with the handle and its new ELO rating.
                    # Also append the details to the 'record_details' list.
                    if i not in latest_elo_score['Discord Handle'].values:
                        new_row = {'Discord Handle':i, 'mean':new_ratings[j][0].mu,'variance':new_ratings[j][0].sigma}
                        new_row_df = pd.DataFrame(new_row, index=[0])  # Convert the dictionary to a DataFrame
                        latest_elo_score = pd.concat([latest_elo_score,new_row_df], ignore_index=True)  # Append the new DataFrame to the existing one
                        # ... (details being added to record_details)
                        record_details.append(i)
                        record_details.append(challenge_df['xMetric Wallet'].iloc[j])
                        record_details.append(challenge_df['Public Result(s)'].iloc[j])
                        record_details.append(challenge_df['Bounty Program Name'].iloc[j])
                        record_details.append(bounty)
                        record_details.append('-')
                        record_details.append(End_Date)
                        record_details.append(challenge_df.iloc[j]['Rank'])
                        record_details.append(processed_data[processed_data['Bounty']==challenge_df.iloc[j]['Bounty']].shape[0])
                        record_details.append(new_ratings[j][0].mu)
                        record_details.append(new_ratings[j][0].sigma)
                        
                        j=j+1
                    # If the 'Discord Handle' is present in the 'latest_elo_score' dataframe,
                    # update its ELO rating in the dataframe. Also append the details to the 'record_details' list.
                    else:
                        latest_elo_score.loc[latest_elo_score['Discord Handle'] == i, 'mean'] = new_ratings[j][0].mu
                        latest_elo_score.loc[latest_elo_score['Discord Handle'] == i, 'variance'] = new_ratings[j][0].sigma

                        # ... (details being added to record_details)
                        record_details.append(i)
                        record_details.append(challenge_df['xMetric Wallet'].iloc[j])
                        record_details.append(challenge_df['Public Result(s)'].iloc[j])
                        record_details.append(challenge_df['Bounty Program Name'].iloc[j])
                        record_details.append(bounty)
                        record_details.append('-') 
                        record_details.append(End_Date)
                        record_details.append(challenge_df.iloc[j]['Rank'])
                        record_details.append(processed_data[processed_data['Bounty']==challenge_df.iloc[j]['Bounty']].shape[0])
                        record_details.append(new_ratings[j][0].mu)
                        record_details.append(new_ratings[j][0].sigma)
                        j=j+1

                # If there is an error while updating the ELO ratings or appending the details, catch the exception and print out the details.
                except Exception as e:
                    logger.error('Error %s in updating elo_score for challenge: %s', e, bounty)  # Add an error statement
                        
                    j=j+1

                # Append the record details to the 'data_with_rank' list. 
                # This list will later be used to create a dataframe with all the details.
                data_with_rank.append(record_details)
            logger.debug('Successfully processed challenge: %s', bounty)  # Add a debugging statement

            # Save the 'latest_elo_score' dataframe to a CSV file.
            # latest_elo_score.to_csv('latest_elo_score.csv')

            
        else:

            # continue
            logger.debug('Iterating over xMetric Wallet values in challenge_df')  # Add a debugging statement

            # Iterate over each value in the 'xMetric Wallet' column of the 'challenge_df' DataFrame.
            for i in (challenge_df['xMetric Wallet']):

                # If the current value is in the 'Discord Handle' column of the 'latest_elo_score' DataFrame,
                # assign a new rating to the 'elo_score' dictionary for that value.
                # The new rating is based on the 'mean' column of the 'latest_elo_score' DataFrame for the corresponding 'Discord Handle'.
                if i in latest_elo_score['Discord Handle'].values:
                    elo_score[i]=Rating(latest_elo_score[latest_elo_score['Discord Handle']==i]['mean'].values[0])


                # If the current value is not in the 'Discord Handle' column of the 'latest_elo_score' DataFrame,
                # assign a new rating of 100 to the 'elo_score' dictionary for that value.
                else:
                    elo_score[i]=Rating(100)



            # Initialize an empty list 'rating_tuple'
            rating_tuple = []

            # Iterate over each item in the 'elo_score' dictionary
            for j in elo_score:
                
                # For each item in the dictionary, create a Rating object using its value
                rating = Rating(elo_score[j])

                # Add the Rating object to the 'rating_tuple' list as a single-item tuple
                rating_tuple.append((rating,))


            # Attempt to calculate new ratings based on the 'rating_tuple' list and ranks from the 'challenge_df' DataFrame
            try:
                new_ratings = rate(rating_tuple, ranks=challenge_df.Rank)


            # If an exception occurs during the rating calculation,
            except Exception as e:
                logger.error('Error calculating new ratings. Error: %s', e)  # Add an error statement

                # Print the exception message
                # print(e)

                continue


            # Initialize a counter 'j' to 0
            j=0

            # Iterate over keys in the 'elo_score' dictionary
            for i in elo_score.keys():
                
                # Initialize an empty list 'record_details'
                record_details=[]


                try:
                    # Check if current key is not in the 'Discord Handle' column of the 'latest_elo_score' DataFrame

                    if i not in latest_elo_score['Discord Handle'].values:

                        
                        # If not, create a new row with 'Discord Handle', 'mean' and 'variance' columns
                        new_row = {'Discord Handle':i, 'mean':new_ratings[j][0].mu,'variance':new_ratings[j][0].sigma}

                        # Append the new row to the 'latest_elo_score' DataFrame
                        new_row_df = pd.DataFrame(new_row, index=[0])  # Convert the dictionary to a DataFrame
                        latest_elo_score = pd.concat([latest_elo_score,new_row_df], ignore_index=True)  # Append the new DataFrame to the existing one


                        # Append relevant details to the 'record_details' list
                        # Append relevant details to the 'record_details' list
                        record_details.append(challenge_df['Discord Handle'].iloc[j])
                        record_details.append(i)
                        record_details.append(challenge_df.iloc[j]['Public Result(s)'])
                        record_details.append(challenge_df.iloc[j]['Bounty Program Name'])
                        record_details.append(challenge_df.iloc[j]['Bounty'])
                        record_details.append(challenge_df.iloc[j]['Challenge_url'])
                        record_details.append(challenge_df.iloc[j]['End Date'])
                        record_details.append(challenge_df.iloc[j]['Rank'])
                        record_details.append(processed_data[processed_data['Bounty']==challenge_df.iloc[j]['Bounty']].shape[0])
                        record_details.append(new_ratings[j][0].mu)
                        record_details.append(new_ratings[j][0].sigma)
                        j=j+1

                    
                    else:
                        # If current key is in the 'Discord Handle' column of the 'latest_elo_score' DataFrame,
                        # update the 'mean' and 'variance' of the corresponding row with the new ratings
                        latest_elo_score.loc[latest_elo_score['Discord Handle'] == i, 'mean'] = new_ratings[j][0].mu
                        latest_elo_score.loc[latest_elo_score['Discord Handle'] == i, 'variance'] = new_ratings[j][0].sigma

                        # Append relevant details to the 'record_details' list
                        # Append relevant details to the 'record_details' list
                        record_details.append(challenge_df['Discord Handle'].iloc[j])
                        record_details.append(i)
                        record_details.append(challenge_df.iloc[j]['Public Result(s)'])
                        record_details.append(challenge_df.iloc[j]['Bounty Program Name'])
                        record_details.append(challenge_df.iloc[j]['Bounty'])
                        record_details.append(challenge_df.iloc[j]['Challenge_url'])
                        record_details.append(challenge_df.iloc[j]['End Date'])
                        record_details.append(challenge_df.iloc[j]['Rank'])
                        record_details.append(processed_data[processed_data['Bounty']==challenge_df.iloc[j]['Bounty']].shape[0])
                        record_details.append(new_ratings[j][0].mu)
                        record_details.append(new_ratings[j][0].sigma)
                        j=j+1



                # Catch any exceptions that occur in the try block
                except Exception as e:
                    # Print the exception message along with a custom message and the challenge URL
                    logger.error('Error processing record details. Error: %s', e)  # Add an error statement

                    # Skip the current iteration and continue with the next one
                    continue

                # If no exceptions occur, append the 'record_details' list to the 'data_with_rank' list
                data_with_rank.append(record_details)

            # Save the 'latest_elo_score' DataFrame to a CSV file
        logger.debug('Successfully processed record details of challenge %s',bounty)  # Add a debugging statement
        latest_elo_score.to_csv('latest_elo_score.csv')
        logger.info('Successfully saved latest_elo_score.csv')  # Add an info statement
        

    logger.debug('Converting data_with_rank list into DataFrame')  # Add a debugging statement
    # Convert the 'data_with_rank' list into a pandas DataFrame.
    Final_data=pd.DataFrame(data_with_rank)

    try:
    # Rename the columns of the DataFrame.
        Final_data.columns=['Discord Handle', 'Address', 'Dashboard link', 'Project', 'Challenge', 'Challenge_link', 'Date', 'Rank','Number of Submissions','Score', 'Variance']
        Final_data['Date'] = pd.to_datetime(Final_data['Date'],dayfirst=True) 
        Final_data['Date'] = Final_data['Date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        # Final_data['Date_str'] = pd.to_datetime(Final_data['Date_str'],format="%d-%m-%Y %H:%M:%S",dayfirst=True) 
        st.dataframe(Final_data[Final_data['Discord Handle']=='sandesh#8645'].sort_values(by='Date',ascending=True))
        # Final_data['Date']=Final_data['Date_str']

    except Exception as e:
        logger.error('Error renaming DataFrame columns. Error: %s', e)  # Add an error statement
        st.write(e)
        pass

    # Save the DataFrame to a CSV file.
    logger.debug('Saving DataFrame to CSV file: true_skill_sheet.csv')  # Add a debugging statement
    Final_data.to_csv('true_skill_sheet.csv')
    logger.info('Successfully saved DataFrame to CSV file: true_skill_sheet.csv')  # Add an info statement

    latest_score_csv = Final_data.sort_values('Date').groupby('Discord Handle',as_index=False)['Score'].last()
    latest_score_csv=pd.DataFrame(latest_score_csv)
    latest_score_csv.columns=['Discord Handle','Score']

    logger.debug('Saving DataFrame to CSV file: latest_score.csv')  # Add a debugging statement
    latest_score_csv.to_csv('latest_score.csv')
    logger.info('Successfully saved DataFrame to CSV file: latest_score.csv')  # Add an info statement

    individual_graph=Final_data[Final_data['Discord Handle']==name]
    



    c1,c2=st.columns((40,60))




    # Convert 'Date' column to datetime if it's not already
    individual_graph['Date'] = pd.to_datetime(individual_graph['Date'])

    # Sort data by date
    individual_graph = individual_graph.sort_values(by='Date')

    # Set 'Date' as index
    individual_graph.set_index('Date', inplace=True)

    # Smooth the 'Score' column with a rolling window operation (mean over window of size N)
    N = 1 # You can change the window size as needed
    individual_graph['Score_smooth'] = individual_graph['Score'].round(0).rolling(window=N).mean()

    # Reset the index
    individual_graph.reset_index(inplace=True)
    

    if individual_graph.shape[0]==0:
        c2.markdown( 
                """
                <div class="card text-white bg-secondary mb-" style="margin:1rem;" >
                <div class="card-header"></div>
                <div class="card-body">
                <h3 class="card-title">NO SCORE</h3>
                <p class="card-text">   
                """, unsafe_allow_html=True
                )
    else:
        fig = go.Figure(go.Indicator(   
            mode = "number",
            value = individual_graph.shape[0],
            number = {'prefix': " ", 'font': {'size': 80}},
            # delta = {'position': "top", 'reference': individual_graph['Score'].iloc[1], 'font': {'size': 24}},
            title = {"text": "<b>Number of dashboards</b>", 'font': {'size': 30}},
            
            domain = {'x': [0, 1], 'y': [0, 1]}))


        fig.update_layout({
            # 'plot_bgcolor': 'rgba(100, 0, 0, 0)',
            # 'paper_bgcolor': 'rgba(215,215,215,255)',
            'height': 200, 
            'width': 400,
            'font': { 'size': 28}  
        })

        c1.plotly_chart(fig, use_container_width=True)


        # Now plot the smoothed data
        individual_graph_smooth=px.line(
                                        individual_graph, x='Date', y='Score_smooth',
                                        hover_data=['Date','Score','Challenge','Rank']
                                        )
        
        individual_graph_smooth.update_yaxes(range=[Final_data['Score'].min(), Final_data['Score'].max()])
        individual_graph_smooth.update_traces(mode='lines+markers',marker=dict(size=6))
        # individual_graph_smooth.update_layout({'plot_bgcolor': 'rgba(100, 0, 0, 0)','paper_bgcolor': 'rgba(25,25,25255)',})
        
        individual_graph_smooth.update_layout(
        title="True Skill of : {}".format(name),
        xaxis_title="Date",
        yaxis_title="Score",
        font=dict(
            color="White"
        ))
        c2.plotly_chart(individual_graph_smooth, use_container_width=True)

        


    individual_graph=individual_graph.sort_values(by='Date',ascending=False)
    try:
        
        Latest_score = individual_graph['Score'].iloc[0]

        fig = go.Figure(go.Indicator(   
            mode = "number+delta",
            value = Latest_score,
            number = {'prefix': " ", 'font': {'size': 80}},
            delta = {'position': "top", 'reference': individual_graph['Score'].iloc[1], 'font': {'size': 24}},
            title = {"text": "<b>Current True skill</b>", 'font': {'size': 30}},
            
            domain = {'x': [0, 1], 'y': [0, 1]}))


        fig.update_layout({
            # 'plot_bgcolor': 'rgba(100, 0, 0, 0)',
            # 'paper_bgcolor': 'rgba(215,215,215,255)',
            'height': 200, 
            'width': 400,
            'font': { 'size': 28}  
        })

        c1.plotly_chart(fig, use_container_width=True)


    except :
        try: 
            
            Latest_score = individual_graph['Score'].iloc[0]



            fig = go.Figure(go.Indicator(
                mode = "number",
                value = Latest_score,
                number = {'prefix': " ", 'font': {'size': 80}},
                # delta = {'position': "top", 'reference': individual_graph['Score'].iloc[1], 'font': {'size': 24}},
                title = {"text": "<b>Current True skill</b>", 'font': {'size': 30}},
                domain = {'x': [0, 1], 'y': [0, 1]}))


            fig.update_layout({
                # 'plot_bgcolor': 'rgba(100, 0, 0, 0)',
                # 'paper_bgcolor': 'rgba(215,215,215,255)',
                'height': 200, 
                'width': 400,
                'font': { 'size': 28}  
            })

            c1.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            c1.markdown( 
                """
                <div class="card text-white bg-secondary mb-" style="margin:1rem;" >
                <div class="card-header"></div>
                <div class="card-body">
                <h3 class="card-title">NO SCORE</h3>
                <p class="card-text">   
                """, unsafe_allow_html=True
                )
            
    c1,c2=st.columns((40,60))

    if individual_graph.shape[0]==0:
        c2.markdown( 
                """
                <div class="card text-white bg-secondary mb-" style="margin:1rem;" >
                <div class="card-header"></div>
                <div class="card-body">
                <h3 class="card-title">NO SCORE</h3>
                <p class="card-text">   
                """, unsafe_allow_html=True
                )
        c1.markdown( 
                """
                <div class="card text-white bg-secondary mb-" style="margin:1rem;" >
                <div class="card-header"></div>
                <div class="card-body">
                <h3 class="card-title">NO SCORE</h3>
                <p class="card-text">   
                """, unsafe_allow_html=True
                )
    else:
        
        # Count the frequency of each bin

        # Define the order of the categories
        cat_type = CategoricalDtype(categories=['Above 10', 'Top 10', 'Top 5', 'Top 3'], ordered=True)

        individual_graph_rank_counts = pd.DataFrame([])
        individual_graph_rank_counts['Rank'] = pd.cut(individual_graph['Rank'], bins=[0,3,5,10,1000], labels=['Top 3', 'Top 5', 'Top 10', 'Above 10'])

        # Convert the 'Rank' column to the categorical type with the specified order
        individual_graph_rank_counts['Rank'] = individual_graph_rank_counts['Rank'].astype(cat_type)

        individual_graph_rank_counts = individual_graph_rank_counts['Rank'].value_counts().sort_index().reset_index()
        individual_graph_rank_counts.columns = ['bin', 'count']
        individual_graph_rank_counts_bar_graph=px.bar(individual_graph_rank_counts, y='bin', x='count', orientation='h',labels={'bin':'Bin', 'count':'Frequency'}, title='Frequency of Rank')
        # individual_graph_rank_counts_bar_graph.update_layout({'plot_bgcolor': 'rgba(100, 0, 0, 0)','paper_bgcolor': 'rgba(25,25,25,255)',})
        individual_graph_rank_counts_bar_graph.update_layout(
        title="Frequency of Rank : {}".format(name),
        xaxis_title="Count",
        yaxis_title="Rank",
        font=dict(
            color="White"
        ))
        c1.plotly_chart(individual_graph_rank_counts_bar_graph, use_container_width=True)
        
        # Count the frequency of each bin

        individual_graph_scatter=px.scatter(
                                            individual_graph, x='Date', y='Rank', color='Rank', 
                                            color_continuous_scale='RdYlGn_r',
                                            color_continuous_midpoint=individual_graph['Rank'].max(),
                                            hover_data=['Date','Score','Challenge','Rank']
                                            )  # This line sets the midpoint of the color scale to the maximum rank)
        individual_graph_scatter.update_traces(marker=dict(size=12))  # This line sets the size of the markers to 12
        individual_graph_scatter.update_layout({'yaxis': {'autorange': "reversed"}})
        individual_graph_scatter.update_layout(
        title="Ranking".format(name),
        xaxis_title="Date",
        yaxis_title="Rank",
        font=dict(
            color="White"
        ))
        c2.plotly_chart(individual_graph_scatter, use_container_width=True)

    individual_data_notion=processed_data[(processed_data['Discord Handle']==name) & (processed_data['Challenge_url']==processed_data['Bounty'])]

    black_marks = pd.read_csv('black_marks.csv')
    black_marks=black_marks[['Discord Handle','Bounty','Public Result(s)','Created At','Notes']]
    black_marks=black_marks[black_marks['Discord Handle']==name]
    if black_marks.shape[0]==0:
        pass 
    else:
        st.header('Black marks')

        st.dataframe(black_marks)
    
    try:
        st.header('Network')

        badges=pd.read_json('https://api.flipsidecrypto.com/api/v2/queries/6f2f3f93-2dff-4c27-a5e4-e840cf2d950d/data/latest')
        mint=badges[badges['EVENT_TYPE']=='mint']
        others=badges[badges['EVENT_TYPE']!='mint']
        mint['concat']=mint['NFT_TO_ADDRESS']+mint['NFT_ADDRESS']+(mint['TOKENID'].apply(lambda x: str(x)))+(mint['ERC1155_VALUE'].apply(lambda x: str(x)))
        others['concat']=others['NFT_FROM_ADDRESS']+others['NFT_ADDRESS']+(others['TOKENID'].apply(lambda x: str(x)))+(others['ERC1155_VALUE'].apply(lambda x: str(x)))
        holders=mint[~mint['concat'].isin(others['concat'])]
        analyst_address=discord_handles[discord_handles['Discord Handle']==name]['xMetric Wallet']
        # Define your mapping dictionary
        mapping_dict = {
            '0xa331500a57c7aaa9c5912d2415f44b399d1c6bc3': 'Seleçao',
            '0xe76645b8b79f2d9147219fa655a4fa96b9f08ffc': 'Eager Beavers',
            '0xc97bec159db7fc28b492c95a4dfd9f2d150d8744': 'Samurai',
            '0x83c923ed97a14271610b8125317a85f469a18662':'Pine',
            '0xd21c9f8b60a94d60ceb305a883e479ec14590970':'Solar Core',
            '0xaee9f539fbae23fa4d11641f39d5e9293e8aecd7':'The Watchmen',
            '0x1804537ec4647be7af16b4a4ec63c3e517a2ee6b': 'Lamoka'
            # Add more mappings as needed
        }

        # Create a new column by mapping the values of 'column_to_map'
        holders['network'] = holders['NFT_ADDRESS'].map(mapping_dict)
        networks_part_of=(holders[holders['NFT_TO_ADDRESS']==analyst_address.values[0]]['network'].unique())
        st.dataframe(networks_part_of,use_container_width=True)
    except:
        pass
with leader_board:
    

    # Assuming df is your DataFrame, 'name' is your name column, and 'date' is your date column


    # Get the latest record for each name
    latest_records = Final_data.loc[Final_data.groupby('Discord Handle')['Date'].idxmax()]

    
    latest_records=(latest_records.sort_values(by='Score',ascending=False).head(10))
    latest_records['Rank']=latest_records['Score'].rank(ascending=False)
    latest_records=latest_records[['Rank','Discord Handle','Address','Date','Score']]
    latest_records.columns=['Rank','Discord Handle','Address','Last active on','Score']

    leaderboard_fig=px.bar(latest_records.sort_values(by='Score',ascending=True),x='Score',y='Discord Handle',orientation='h',
                           title='Leaderboard',color='Score',color_continuous_scale='YlGn_r')

    st.plotly_chart(leaderboard_fig,use_container_width=True)


    st.dataframe(latest_records,use_container_width=True,hide_index=True)
with contact: 
    st.header("Contact info")
    st.write(f"""
            Discord : sandesh8645 \n
            Twitter : https://twitter.com/Sandesh_K_12 \n
            Github  : https://github.com/sandeshsk12/true_skill
            """)
with credits:
    st.header("Acknowledgements")

    st.write(f"""
            Shoutout to Neby (aka. Not [moisturized] smol data) for his excellent work! Acknowledging the contributions of individuals like Neby is important as it encourages a collaborative and supportive environment. So, thank you, Neby, for your valuable contributions and for making a positive impact!
            
            Checkout his dashboard and smash some like buttons at https://flipsidecrypto.xyz/LittlerData/metrics-dao-analytics-marketplace-through-the-magnifier-1A7FAJ """
            )
    st.write(f"""
             Resources : \n
             1. True skill package : https://trueskill.org/ \n 
             2. True skill theory : https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
             """)



