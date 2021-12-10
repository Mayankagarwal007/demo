import eel
import concurrent.futures
import math
import statistics
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import date, timedelta
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

eel.init('web')

result = ""
humid = []
rainfall = []
temperature = [] 

MAX_THREADS = 30

#building the ML MOdel for prediction
data = pd.read_csv('data.csv')
x = data.drop(['label'],axis=1)
x = x.values
km = KMeans(n_clusters = 4, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns = {0 : 'cluster'})
y = data['label']
x = data.drop(['label'], axis=1)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0)
model = LogisticRegression()
print("Build ho rha hai")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

@eel.expose
def say_hello_py(x):
    for i in range(0,len(x)):
        if(x[i]==':'):
            p1 = i
        if(x[i]==','):
            p2 = i
            break
    lng = x[p1+1:p2]
    for i in range(p2,len(x)):
        if(x[i]==':'):
            p1 = i
            break
    lat = x[p1+1:len(x)-1]
    compute_urls(lat,lng)

def predict(x,y,z):
    prediction = model.predict(np.array([[x,y,z]]))
    global result
    result = prediction[0]

def download_url(quote_page):
    page = urlopen(quote_page)
    soup = BeautifulSoup(page, "html.parser")
    humidity_box = soup.find(attrs={"class": "humidity"}).find(attrs={"class": "num swip"})
    rain_box = soup.find(attrs={"class": "precipAccum swap"}).find(attrs={"class": "num swip"})
    temp_box = soup.find(attrs={"class": "temperature"}).find(attrs={"class": "num"})
    global humid
    global rainfall
    global temperature
    a = float(humidity_box.text)
    b = float(rain_box.text)
    c = float(temp_box.text)
    if np.isnan(a) == False:
        humid.append(a)
    if np.isnan(b) == False:
        rainfall.append(b)
    if np.isnan(c) == False:
        temperature.append(c)
    

def download_stories(story_urls):
    threads = min(MAX_THREADS, len(story_urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_url, story_urls)

def compute_urls(lat,lng):
    urls = []
    global humid
    global temperature
    global rainfall
    global result
    humid.clear()
    temperature.clear()
    rainfall.clear()
    for i in range(365):
        d = date.today() - timedelta(i)
        quote_page = "https://darksky.net/details/" + lat + ',' + lng + '/' + str(d) + "/us12/en"
        urls.append(quote_page)
    download_stories(urls)
    humidity_index = statistics.mean(humid)
    rainfall_index = statistics.mean(rainfall) * 1000
    temperature_index = statistics.mean(temperature)
    temperature_index = (temperature_index - 32) * (5/9) 
    predict(temperature_index,humidity_index,rainfall_index)
    result = result + " " + str(temperature_index) + " " + str(humidity_index) + " " + str(rainfall_index) + " "
    # print("Humidity = ", humidity_index) 
    # print("Tempearture = ", temperature_index)
    # print("Rainfall = ", rainfall_index) 
    # print(z[z['cluster'] == 0]['label'].unique())
    # print(z[z['cluster'] == 1]['label'].unique())
    # print(z[z['cluster'] == 2]['label'].unique())
    # print(z[z['cluster'] == 3]['label'].unique())

@eel.expose
def say_hello_js():
    return result

eel.start('index.html', size=(300, 200))
