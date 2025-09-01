import pandas as pd
import random

# Nilai kemungkinan untuk tiap atribut
outlooks = ['Sunny', 'Cloudy', 'Rainy']
temperatures = ['Hot', 'Mild', 'Cool']
humidities = ['High', 'Normal']
windy_vals = [True, False]

# Fungsi sederhana untuk menentukan nilai PLAY
def determine_play(outlook, humidity, windy):
    if outlook == 'Sunny' and humidity == 'High':
        return 'NO'
    elif outlook == 'Rainy' and windy:
        return 'NO'
    else:
        return 'YES'

# Generate 100 baris data dummy
data = []
for i in range(1, 101):     #ubah ini aja buat generate n-baris
    outlook = random.choice(outlooks)
    temperature = random.choice(temperatures)
    humidity = random.choice(humidities)
    windy = random.choice(windy_vals)
    play = determine_play(outlook, humidity, windy)
    data.append([i, outlook, temperature, humidity, windy, play])

# Buat DataFrame dan simpan ke CSV
df = pd.DataFrame(data, columns=["ID", "OUTLOOK", "TEMPERATUR", "HUMIDITY", "WINDY", "PLAY"])
df.to_csv("dummy_weather_data_100.csv", index=False)

print("File berhasil disimpan sebagai 'dummy_weather_data_100.csv'")
