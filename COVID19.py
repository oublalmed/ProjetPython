#import des librairies l'environnement
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


import plotly.express as px
import folium

#Importation des données

dt = pd.read_csv('C:/Users/Hp/Desktop/non complete/Projet Python/output.csv')
data = pd.read_csv('C:/Users/Hp/Desktop/non complete/Projet Python/covid_19_clean_complete.csv')

#spécifier le style de la table
dt.style.background_gradient(cmap="Blues", subset=['Confirmed', 'Active'])\
            .background_gradient(cmap="Greens", subset=['Recovered'])\
            .background_gradient(cmap="Reds", subset=['Deaths'])

# les variables x et Y

feature_names = ['Confirmed', 'Active', 'Deaths','Recovered']
X = dt[feature_names]
y = dt['Classes']

                 # scatter matrix

cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('COVID19_scatter_matrix')


                 #train et test des données

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

                 # Regression logistique

model = LogisticRegression(C=1.0, multi_class='auto', solver='liblinear')
model.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(model.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(model.score(X_test, y_test)))



# map

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(data)):
    folium.Circle(
        location=[data.iloc[i]['Lat'], data.iloc[i]['Long']],
        color='crimson',
        tooltip =   '<li><bold>Country : '+str(data.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(data.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(data.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(data.iloc[i]['Deaths']),
        radius=int(data.iloc[i]['Confirmed'])**1.1).add_to(m)
m


# graphe represente la somme des cas (Deaths and Recovred) par Date

temp = data.groupby('Date')['Recovered', 'Deaths'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=800,
             title='Cases over time', color_discrete_sequence = ['green', 'red'])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()


# graphe represente les Region par les (Deaths and Confirmed)

fig = px.scatter(dt.sort_values('Deaths', ascending=False).iloc[:15, :], 
                 x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=800,
                 text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed')
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
