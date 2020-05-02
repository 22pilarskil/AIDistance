from django.db import models
import pyrebase
# Create your models here.
config = {
    "apiKey": "AIzaSyD9HX-My-yXdQughjTil58BaIu-HAQzkIA",
    "authDomain": "ai-distance.firebaseapp.com",
    "databaseURL": "https://ai-distance.firebaseio.com",
    "storageBucket": "ai-distance.appspot.com",
    "serviceAccount": "static/json/serviceAccountCredentials.json"
  }
firebase=pyrebase.initialize_app(config)

db = firebase.database()

def addUser(username):
    db.child("Users").child(username).child("username").set(username) 

def addLocation(details):
    db.child("Locations").child(details["location"]).child("name").set(details["location"]) 