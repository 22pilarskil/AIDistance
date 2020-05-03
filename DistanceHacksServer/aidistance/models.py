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
    db.child("Locations").child(details["location"]).child("Address").set(details["address"]) 
    db.child("Locations").child(details["location"]).child("SquareFeet").set(details["squarefeet"]) 
    db.child("Locations").child(details["location"]).child("People").set(details["people"]) 
    db.child("Locations").child(details["location"]).child("safe").set(details["safe"]) 

def setNumberofPeople(details):
    db.child("Locations").child(details["location"]).child("People").set(details["people"]) 
    if int(db.child("Locations").child(details["location"]).get().val()['SquareFeet'])/int(details["people"]) < 6:
        db.child("Locations").child(details["location"]).child("safe").set("Not Safe") 
    else:
        db.child("Locations").child(details["location"]).child("safe").set("Safe") 


def getLocations():
    locations = db.child("Locations").get().val()
    return locations
