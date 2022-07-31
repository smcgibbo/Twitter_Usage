# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns
import csv
from functools import reduce # Python3
from collections import Counter
import Levenshtein as lev
from tqdm import tqdm
#from Levenshtein import distance as lev

def make_colourmap():
    # Tableau 20 color palette for demonstration
    colors = [(211, 213, 217), (167, 171, 179), (123, 129, 141), (80, 87, 103), (36, 46, 66),(132,200,255), (31, 162, 255)]
    colors = [(36, 46, 66),(80, 87, 103), (123, 129, 141),(167, 171, 179),(211, 213, 217),(214, 237, 255), (174, 218, 255), (132, 200, 255), (91, 182, 255), (31, 162, 255)]
    # Conversion to [0.0 - 1.0] from [0.0 - 255.0]
    colors = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in colors]
    cmap = ListedColormap(colors)
    return(cmap)

start = timer()
# Creating list to append tweet data 
tweets_list1 = []
hashtag_list1 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
user_list = ['@VolunteerTO','@o_n_n','@EndeavourVCN', '@180Degrees','@ImagineCanada','@canadahelps', '@VolunteerCanada']
#user_list = ['@180Degrees']

#with open('twitter_list.csv', newline='') as f:
#    reader = csv.reader(f)
#    l = list(reader)
#user_list = reduce(lambda x,y :x+y ,l)
print(user_list)

def hamming_distance(chaine1, chaine2):
    print(tuple( zip(chaine1, chaine2)))
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))



for user in tqdm(user_list):
    user_hastags = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+user+' since:2021-01-01 until:2022-01-01').get_items()):# include:nativeretweets
        #if i>5: #number of tweets you want to scrape
        #    break
        try:
            mentioned = len(tweet.mentionedUsers)
        except TypeError:
            mentioned = 0  # or whatever you want to do
        tweets_list1.append([tweet.date,tweet.user.username, tweet.content, tweet.tcooutlinks, mentioned,
                             tweet.replyCount, tweet.retweetCount,tweet.likeCount, tweet.quoteCount])#declare the attributes to be returned
        l = list(set([i[1:] for i in  tweet.content.split() if i.startswith("#")]))
        
        for j in range(len(l)):
            string = l[j]
            string = string.replace(",", "")
            string = string.replace(".", "")
            string = string.replace(":", "")
            string = string.replace(")", "")
            string = string.replace("!", "")
            string = string.replace("?", "")
            string = string.replace("\"", "")
            string = string.replace("\'", "")
            l[j] = string.lower()
            if l[j].lower() == 'earthday':
                print(tweet.user)
                print(tweet.content)
        user_hastags.append(l)
    u_set =  list([item for sublist in user_hastags for item in sublist])#list(set([item for sublist in user_hastags for item in sublist]))
    #print(u_set)
    h = u_set
    for k in range(len(h)):
        for j in range(k+1,len(h)):
            dist =  lev.distance(h[k], h[j])
            #print(h[i], h[j], dist)
            if dist > 0 and dist <=1:
                h[k] = sorted([h[k], h[j]], key=len)[1]
                h[j] = sorted([h[k], h[j]], key=len)[1]
    hashtag_list1.append(list(set(h)))#
    #print(user, timer()-start)


#print(hashtag_list1)
h = [item for sublist in hashtag_list1 for item in sublist]

for i in range(len(h)):
    for j in range(i+1,len(h)):
        dist =  lev.distance(h[i], h[j])
        #print(h[i], h[j], dist)
        if dist > 0 and dist <=1 and len(h[i])>2 and len(h[j]) >2:
            if h[j] == 'ild2021' or h[i] == 'ild2021':
                print(h[j], h[i])
            #print(sorted([h[i], h[j]], key=len)[0])
            h[i] = sorted([h[i], h[j]], key=len)[0]
            h[j] = sorted([h[i], h[j]], key=len)[0]

#print(h)
l = []
for i in range(len(h)):
    if len(h[i]) != 0:
        l.append(h[i])



        
h_counts = Counter(h)
df = pd.DataFrame.from_dict(h_counts, orient='index', columns = ['count'])
'''print(len(df))
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(subset = ["column2"], inplace=True)'''
print(len(df))
top = df.nlargest(30, 'count')
print(top)
top.plot(kind='bar')
plt.show()
# Creating a dataframe from the tweets list above 
#tweets_df = pd.DataFrame(tweets_list1, columns=['Datetime','User', 'Text', 'Outlink', 'In Tweet Mentions', 'Replies','Retweets', 'Likes', 'Quotes'])
#tweets_df.to_csv('benchmark_data.csv', index = False)
#print(tweets_df)
#print(tweets_df.groupby(tweets_df['User']).size())
#print(tweets_df.groupby([tweets_df['User'], tweets_df['Retweet']]).size())
#print(tweets_df.groupby(tweets_df['User'])['Likes'].mean())
#print(tweets_df.groupby(tweets_df['User'])['Likes'].std())
#print(tweets_df.groupby(tweets_df['User'])['Retweets'].mean())
#print(tweets_df.groupby(tweets_df['User'])['Replies'].mean())
#print(tweets_df.groupby([tweets_df['User'], tweets_df['In Tweet Mentions']]).size())
#user_df = tweets_df.groupby([tweets_df['User'], tweets_df['In Tweet Mentions']])['Likes'].sum()
#print(user_df)
'''
#tweets_df = pd.read_csv('NFP_data.csv', header = 0, parse_dates = [1])
tweets_df = pd.read_csv('benchmark_data.csv', header = 0, parse_dates = [0])
print(len(tweets_df))
#print(tweets_df)
#print(tweets_df['Datetime'].dt.tz)

tweets_df['Datetime'] = tweets_df['Datetime'].dt.tz_convert('US/Eastern')
#print(tweets_df)


#average number of posts _________________________
user_df = tweets_df.groupby(tweets_df['User']).size()#histogram of tweets by user group
print(user_df)
#print(tweets_df.groupby(tweets_df['User'])['Likes'].mean())
#print(tweets_df.groupby(tweets_df['User'])['Likes'].std())
#print(tweets_df.groupby(tweets_df['User'])['Retweets'].mean())
#print(tweets_df.groupby(tweets_df['User'])['Replies'].mean())
print(tweets_df.groupby(tweets_df['User'])['In Tweet Mentions'].mean())
user_df = tweets_df.groupby([tweets_df['User'], tweets_df['In Tweet Mentions']])['Likes'].sum()

print(user_df)
'''
'''
#Heatmap__________________________________________
#piv = pd.pivot_table(df_heat.reset_index(name='count'), values="count",index=["Hours"], columns=["Weekday"], fill_value=0)
df_heat = tweets_df.groupby( [ tweets_df['Datetime'].dt.hour, tweets_df['Datetime'].dt.weekday]).size()#.reset_index(name='count')
df_heat.index.rename(['Hour', 'Weekday'], inplace = True)
df_heat.reset_index(name='count')
#print(df_heat.reset_index(name='count'))
piv = pd.pivot_table(df_heat.reset_index(name='count'), values="count",index=["Hour"], columns=["Weekday"], fill_value=0).transpose()
print(piv)

week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.figure(figsize=(15, 7))
ax = sns.heatmap(piv, cmap = make_colourmap(), cbar = True, yticklabels = week, square=True, cbar_kws={"shrink": 0.68})
ax.set_xlim(7, 23)
#plt.setp(ax.xaxis.get_majorticklabels(), rotation=90 )
plt.tight_layout()
end = timer()
#print('\n Time in seconds:', end - start, 'Time in minutes: ', (end - start)/60.)
ax.set_xlabel('Hour (EST)', fontsize=14)
ax.set_ylabel('Day of Week', fontsize=14)

plt.show()
#____________________________________________________
'''
'''
plt.figure(figsize=(10, 5))
ax = (tweets_df['Datetime'].groupby(tweets_df['Datetime'].dt.hour, tweets_df['Datetime'].dt.weekday).count()).plot(kind="bar", color='#494949')
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Number of tweets")
#positions = (0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
#labels = (19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
#plt.xticks(positions, labels)
end = timer()
print('\n Time in seconds:', end - start, 'Time in minutes: ', (end - start)/60.)
plt.show()
'''
#correlation with mention amount, or groupby mentions and compare 0 vs. more than 0 for likes, retweets, comments

'''
# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#GivingTuesday since:2021-011-15 until:2021-12-02').get_items()):
    if i>10:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
print(tweets_df2)
'''

