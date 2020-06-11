import pandas as pd
import os
import pickle
import math

models = []
zipcode = pd.DataFrame()
US_cities = pd.DataFrame()
zip_sex_pop = pd.DataFrame()

def load_models(path):
    global models, zipcode, US_cities, zip_sex_pop
    files = os.listdir(path)
    models = []
    for file in files:
        with open(path +'\\'+ file, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
            models.append(model)

    zipcode = pd.read_csv('./application/US.txt', sep='\t')
    US_cities = pd.read_csv('./application/US_cities.csv')
    zip_sex_pop = pd.read_csv('./application/zip_sex_pop.csv')

# добавляет координаты почтовых отделений
def add_zip_coordinates(df_input, zipcode):
    df_output = df_input.copy()
    df_output = df_output.merge(zipcode[['postal_code', 'code1', 'latitude', 'longitude']],
                                            how='inner',
                                            left_on='zipcode',
                                            right_on='postal_code')
    df_output.drop(['postal_code', 'code1'], axis=1, inplace=True)
    df_output.rename(columns={'latitude': 'zip_latitude', 'longitude': 'zip_longitude'}, inplace=True)
    return df_output

# добавляет координаты центра городов
def add_city_coordinates(df_input, US_cities):
    df_output = df_input.copy()
    df_output['city'] = df_output['city'].apply(lambda x: x.lower())
    df_output['state'] = df_output['state'].apply(lambda x: x.lower())
    US_cities['City'] = US_cities['City'].apply(lambda x: x.lower())
    US_cities['Region'] = US_cities['Region'].apply(lambda x: x.lower())
    df_output = df_output.merge(US_cities[['City', 'Region', 'Latitude', 'Longitude']],
                                            how='inner',
                                            left_on=['city', 'state'],
                                            right_on=['City', 'Region'])
    df_output.rename(columns={'Latitude': 'city_latitude', 'Longitude': 'city_longitude'}, inplace=True)
    df_output.drop(['City', 'Region'], axis=1, inplace=True)
    return df_output

# функция считает растояние и азимут между двумя точками
# код взят здесь https://pastebin.com/PHeWmiEN
def dist_azimut(llat1, llong1, llat2, llong2):
    #pi - число pi, rad - радиус сферы (Земли)
    rad = 6372795
    #координаты двух точек
    #в радианах
    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.
    #косинусы и синусы широт и разницы долгот
    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)
    #вычисления длины большого круга
    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)
    dist = round(ad*rad) # в метрах
    #вычисление начального азимута
    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))
    if (x < 0):
        z = z+180.
    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi)))) # в радианах
    angledeg = round((anglerad2*180.)/math.pi, 2) # в градусах
    #row['distance'] = dist
    #row['azimut'] = angledeg
    return dist, angledeg

def distCentePost(row):
    dist, az = dist_azimut(float(row['city_latitude']),
                           float(row['city_longitude']),
                           float(row['zip_latitude']),
                           float(row['zip_longitude']))
    row['centreToPost'] = dist
    row['postAzimuth'] = az
    return row

def get_prediction(features):
    features = add_zip_coordinates(features, zipcode)
    features = add_city_coordinates(features, US_cities)
    features = features.merge(zip_sex_pop, how='inner', left_on='zipcode', right_on='zipCode')
    features.drop('zipCode', axis=1, inplace=True)
    features = features.apply(lambda row: distCentePost(row), axis=1)
    features.drop(['city', 'state', 'zipcode', 'zip_latitude', 'zip_longitude', 'city_latitude',
       'city_longitude'], axis=1, inplace=True)

    sqft = features['sqft'][0]

    predictions = pd.DataFrame()
    for i, model in enumerate(models):
        predictions['pred_'+str(i)] = model.predict(features).round(0)
    predictions['blend'] = round((predictions.sum(axis=1))/len(predictions.columns))
    price = predictions['blend'][0]*sqft
    return price
