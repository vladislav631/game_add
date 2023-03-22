#!/usr/bin/env python
# coding: utf-8

# ### Задача:
# 
# Сформируйте модель монетизации игрового приложения.
# 
# Многие игры зарабатывают с помощью рекламы. И все они сталкиваются с противоречием:
# 
# * Пользователь разозлится и уйдёт, если начать показывать ему рекламу раньше, чем игра его затянет.
# * Но  чем позже создатели игры включат рекламу, тем меньше они заработают.
# 
# Аналитик помогает бизнесу выбрать оптимальное время для запуска рекламы. Зная расходы на продвижение игры, он может рассчитать её окупаемость при разных сценариях 
# 
# Пока создатели игры планируют показывать её на экране выбором постройки. Помогите им не уйти в минус.

# 1. Предобработка данных

# 2. Исследовательский анализ данных

# * количество игроков, перешедших на 1 уровень победив врага
# * метрики монетизации:
#     * DAU, WAU 
# * график по событиям, включая игроков перешедших на 1 уровень победив врага 
# * график по количеству объектов
# * график по реализованным проектам 
# * построить график по дням, по которому произошел клик по объявлению
# * график для источников, с которых пришел пользователь 

# 3. Статистические гипотезы

# * Проверьте гипотезу различия времени прохождения уровня между пользователями, которые заканчивают уровень через реализацию проекта, и пользователями, которые заканчивают уровень победой над другим игроком.
# 
# Сформулируйте и проверьте статистическую гипотезу относительно представленных данных:
#    * Проверить различие кто больше приносит денег по кликам - пользователи, которые заканчивают уровень "побеждая врага" или пользователи, которые заканчивают уровень через реализацию проекта 

# 4. Выводы

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)})

import scipy.stats as stats
from scipy import stats as st

import math as mth

import numpy as np

import pandas as pdm
from datetime import datetime,timedelta

from pathlib import Path
import matplotlib.dates as mdates

import math
import cmath


# In[2]:


ad_costs = pd.read_csv('/datasets/ad_costs.csv', sep=',')
user_source = pd.read_csv('/datasets/user_source.csv', sep=',')
game_actions = pd.read_csv('/datasets/game_actions.csv', sep=',')


# In[3]:


ad_costs.info()


# In[4]:


user_source.info()


# In[5]:


game_actions.info()


# In[6]:


ad_costs.duplicated().sum()


# In[7]:


game_actions.duplicated().sum()


# In[8]:


game_actions = game_actions.drop_duplicates()


# In[9]:


user_source.duplicated().sum()


# In[10]:


ad_costs['day'] = pd.to_datetime(ad_costs['day'])


# In[11]:


game_actions.head()


# In[12]:


game_actions['project_type'].unique()


# In[13]:


game_actions['building_type'].unique()


# могу предположить что пропуски в building_type являются следствием того что здания просто напросто непостроены, а project_type следствие того что орбитальная станция не построена

# In[14]:


game_actions['project_type'] = game_actions['project_type'].fillna('unknown')


# In[15]:


game_actions['building_type'] = game_actions['building_type'].fillna('unknown')


# In[16]:


game_actions.info()


# In[17]:


game_actions['time'] = pd.to_datetime(game_actions['event_datetime'])
game_actions.info()


# In[18]:


game_actions.sample(5)


# In[19]:


game_actions['project_type'].unique()


# In[20]:


game_actions['building_type'].unique()


# In[21]:


user_source.head()


# In[22]:


ad_costs.sample(5)


# In[23]:


ad_costs.describe()


# ничего больше интересного в предобработке нет 

# ### Исследовательский анализ данных

# #### количество игроков, перешедших на 1 уровень победив врага

# In[24]:


game_actions['event'].unique()


# In[25]:


project_finished = game_actions.query('event == ("project", "finished_stage_1")')


# In[26]:


project_finished['count'] = project_finished['user_id'].map(project_finished['user_id'].value_counts())
#project_finished['count'] = project_finished.groupby('user_id')['event'].count().reset_index().transform('count')
project_finished


# In[27]:


project_finished['count'] = project_finished['count'].astype(int)


# In[28]:


project_finished['count'].unique()


# In[29]:


project_finished['count'] = ['warrior' if x == 1 else 'builder' for x in project_finished['count']]


# In[30]:


project_finished['count'].unique()


# In[31]:


project_finished.head()


# In[32]:


df1 = pd.merge(game_actions, project_finished, how = 'left')
df1 = df1.dropna()
df1


# #### метрики монетизации:
# * DAU, WAU

# In[33]:


game_actions = game_actions.rename(columns={'event_datetime': 'day'})


# In[34]:


df = pd.merge(game_actions, user_source, how = 'left')

df.info()


# In[35]:


df.head()


# In[36]:


df['day'] = df['day'].astype('datetime64')


# In[37]:


df['week'] = df['day'].dt.week


# In[38]:


dau = df.groupby('day').agg({'user_id': 'nunique'})


# In[39]:


ax_dau = dau.plot()
ax_dau.set_title('Зависимость посещения по дням')
ax_dau.set_xlabel('Дата')
ax_dau.set_ylabel('Посещения')


# видим что все меньше пользователей получает новый уровень 

# In[40]:


wau = df.groupby('week').agg({'user_id': 'nunique'})
wau


# #### график по событиям, включая игроков перешедших на 1 уровень победив врага

# In[41]:


import plotly.graph_objects as go


# In[42]:


pie = df.groupby('event')['user_id'].count().reset_index()

labels = pie.event
values = pie.user_id

fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='label+value+percent'
                  )
fig.update_layout(
    title_text="Проекты",
    width=1000, 
    height=400)
fig.show()


# #### график по реализованным проектам

# In[43]:


pie = df.groupby('project_type')['user_id'].count().reset_index()

labels = pie.project_type
values = pie.user_id

fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='label+value+percent'
                  )
fig.update_layout(
    title_text="Реализованные проекты",
    width=1000, 
    height=400)
fig.show()


# #### построить график по дням, по которому произошел клик по объявлению

# In[44]:


import plotly.express as px


# In[45]:


ad_costs.head()


# In[46]:


fig = px.line(ad_costs.groupby('day')['cost'].sum().reset_index(), x='day', y='cost')
fig.update_layout(
    title_text="график по дням, по которому произошел клик по объявлению (cost)")
fig.show()


# #### график для источников, с которых пришел пользователь

# In[47]:


ad_costs


# In[48]:


pie = user_source.groupby('source')['user_id'].count().reset_index()

labels = pie.source
values = pie.user_id

fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='label+value+percent'
                  )
fig.update_layout(
    title_text="Переходы",
    width=1000, 
    height=400)
fig.show()


# пользователи которые пришли из разных источников 

# Статистические гипотезы
# * Проверьте гипотезу различия времени прохождения уровня между пользователями, которые заканчивают уровень через реализацию проекта, и пользователями, которые заканчивают уровень победой над другим игроком.
# 
# Сформулируйте и проверьте статистическую гипотезу относительно представленных данных:
# 
# * Проверить различие кто больше приносит денег по кликам - пользователи, которые заканчивают уровень "побеждая врага" или пользователи, которые заканчивают уровень через реализацию проекта

# * Н0 - время прохождения уровня между пользователями, которые заканчивают уровень через реализацию проекта, и пользователями, которые заканчивают уровень победой над другим игроком статистически значима
# * Н1 - различие не статистически значимо 

# In[49]:


min_event = game_actions.groupby(['user_id','event'])['time'].min().reset_index()
time_event = game_actions.query("event == 'finished_stage_1'")[['user_id','time']]
event_time = min_event.merge(time_event,on = 'user_id', how = 'inner')
event_time.columns = ['user_id','event','start','finish']
event_time['diff_time_event'] = event_time['finish'] - event_time['start']
event_time = event_time.merge(df1, on ='user_id', how = 'inner')
event_time.head(1)


# In[50]:


alpha = 0.05
results = st.mannwhitneyu(event_time[event_time['count']=='warrior']['diff_time_event'], event_time[event_time['count']=='builder']['diff_time_event'],alternative = 'two-sided')
pvalue = results.pvalue
print('p-значение: ', pvalue)
if (pvalue < alpha):
    print("Отвергаем H0: разница статистически значима")
else:
    print("Не получилось отвергнуть H0: вывод о различии сделать нельзя")


# In[51]:


df1[df1['count']=='warrior']['event_datetime'].head()


# In[52]:


df1[df1['count']=='warrior']['time'].head()


# In[53]:


df2 = df[df['source']=='yandex_direct'].set_index('day')['2020-05-03':'2020-05-09'].nunique()


# In[54]:


df2


# In[55]:


df3 = df[df['source']=='youtube_channel_reklama'].set_index('day')['2020-05-03':'2020-05-09'].nunique()
df3


# In[56]:


count_warrior_youtube = df.query("event == 'finished_stage_1' and source =='youtube_channel_reklama'").count()
count_warrior_youtube.unique()


# In[57]:


count_warrior_yandex = df.query("event == 'finished_stage_1' and source =='yandex_direct'").count()
count_warrior_yandex.unique()


# In[58]:


ad_costs['cost'] = ad_costs['cost'].astype(int)


# In[59]:


yandex_direct = ad_costs['cost'][ad_costs['source']=='yandex_direct'].sum()
youtube_channel_reklama = ad_costs['cost'][ad_costs['source']=='youtube_channel_reklama'].sum()
yandex_direct


# * H0 - нет различий между теми кто прошел уровень придя из yandex_direct и youtube_channel_reklama соответственно 
# * H1 - группы прошедших уровень, которые установили приложения через yandex_direct и youtube_channel_reklama разные 

# In[60]:


alpha=0.05
purchases = np.array([1159,2042])
leads = np.array([2630, 4728])
p1 = purchases[0] / leads[0]
p2 = purchases[1] / leads[1]
combined = (purchases[0] + purchases[1]) / (leads[0] + leads[1])
difference = p1-p2
z_value = difference / math.sqrt(combined * (1 - combined) * (1 / leads[0] + 1 / leads[1]))
distr = st.norm(0,1)
p_value = (1 - distr.cdf(abs(z_value))) * 2
print('p-значение: ', p_value)
if (p_value < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# ### Выводы

# * больше всего приходят из яндекс директа 
# * малое количество игроко (1%) доходят до постройки орбитальной станции 
# * посещения в игру к концу обозреваемого периода падают
# * Ни одна гипотеза не потвердилась - различия между yandex_direct и youtube_channel_reklama  нет, а также время прохождения уровня между пользователями, которые заканчивают уровень через реализацию проекта, и пользователями, которые заканчивают уровень победой над другим игроком не различается
# * из графиков можно сделать вывод что интерес к игре начинает теряться на 8ой день, почти полностью теряется на 21ый день, поэтому предлагаю начинать запускать рекламу в период с 3ьего по 5ый день использования игры

# In[61]:


df1.to_csv(r'C:\Users\vneso\Downloads\file.csv')


# In[ ]:




