import pyttsx3
from os import name
import datetime
import speech_recognition as sr
import PySimpleGUI as sg
import random
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import tkinter as tk
from model import preprocess, predict
import pickle


with open('tokenizer.pickle', 'rb') as encoder:
        tokenizer = pickle.load(encoder)
with open('label_tokenizer.pickle', 'rb') as label_encoder:
        label_tokenizer = pickle.load(label_encoder)


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)




def randomDate():
    start_date = datetime.date(2021, 10, 27)
    end_date = datetime.date(2021, 11, 27)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good morning")
    if hour > 12 and hour <= 24:
        speak("Good evening")
    speak("Iam Zoe, how may I help you?")


query = ''


def takeCommand():
    global query
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.energy_threshold = 3500
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        print(query)
        query = query.lower()
        if query == "":
            speak("Sorry, I didnt quite get that")

        print(f"User said: {query}\n")
    except:
        speak("Sorry, I didnt quite get that")


# machine learning model
# def predict(test):
#     d = {1: "viral infection, flu", 2: "diarrhea", 3: "covid19", 4: "tiredness, take rest", 5: "asthma", 6: "migraine",
#          7: "allergy"}
#     d2 = {"fever": 1, "feverish": 2, "sneezing": 3, "common cold": 4, "stomach ache": 5, "cough": 6, "loose motion": 7,
#           "stomach pain": 8, "nose block": 9, "sore throat": 10, "body ache": 11, "chills": 12, "shivering": 13,
#           "shaking": 14, "loss of taste": 15, "loss of smell": 16, "shortness of breath": 17, "runny nose": 18,
#           "rashes": 19, "headache": 20, "cold":21,}
#     data = pd.read_csv("diseases and symptoms1.csv")
#     X = data.drop(columns=["diseases"])
#     Y = data["diseases"]
#     model = DecisionTreeClassifier()
#     model.fit(X, Y)
#     pred = model.predict([[d2[test]]])
#     return d[pred[0]]


# --------------------------------------------------------


layout = [[sg.Text("Check your symptoms")], [sg.Button("1")], [sg.Text("Book an appointment")], [sg.Button("2")],
          [sg.Text("Show directions")], [sg.Button("3")], [sg.Text("Quit")], [sg.Button("4")]]
window = sg.Window("Bot", layout)
wishme()


def take_symptom():
    speak("What are your symptoms? ")
    takeCommand()
    global query
    query_preprocessed = preprocess(query)
    disease = predict(query_preprocessed, tokenizer, label_tokenizer)
    print(disease)
    speak(f'You may have {disease}')



def Book_appointment():
    speak(f"Your appointment has been booked on: {randomDate()} ")


def Directions():
    speak("Where do you want to go?")
    takeCommand()
    if 'reception' in query:
        speak('Take a left and follow signs to reception')
    if 'pharmacy' in query:
        speak('Take a right and follow signs to pharmacy.')
    if 'emergency' in query:
        speak('Follow signs to trauma center')



window = tk.Tk()
window.geometry("1920x1080")

bg = tk.PhotoImage(file="./logo.png")
my_canvas=tk.Canvas(window,width=1920,height=1080)
my_canvas.pack(fill="both",expand=True)
my_canvas.create_image(0,0,image=bg,anchor='nw')
heading = tk.Label(window,text = "Zoe", padx=30)
heading.place(x=650,y=25)
heading.configure(font = ("Times New Roman", 74, "bold"))



Label = tk.Label(window,text="Hello, This is Zoe your personal healthcare assistant", pady=5, padx = 30)
Label.configure(font = ("Times New Roman", 40, "bold"))
Label.place(x=200,y=250)
btn1 = tk.Button(window,text="To Book an appointment press here",pady=5, command=Book_appointment,bg= '#1d2731' ,fg='white')
btn1.place(x=500,y=400)


btn1.configure(font = ("Times New Roman", 24, "bold"))
btn2 = tk.Button(window,text="To Check your symptoms press here",pady=5, command =take_symptom,bg= '#1d2731',fg='white')
btn2.place(x=500,y=500)
btn2.configure(font = ("Times New Roman", 24, "bold"))
btn3 = tk.Button(window, text="Show directions",pady=5, command=Directions,bg= '#1d2731',fg='white')
btn3.place(x=600,y=600)
btn3.configure(font = ("Times New Roman", 24, "bold"))
btn4 = tk.Button(window,text="Quit", command=window.destroy,pady=5,bg= '#1d2731',fg='white')
btn4.place(x=600,y=700)

btn4.configure(font = ("Times New Roman", 24, "bold"))
window.mainloop()





