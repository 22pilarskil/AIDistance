from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from aidistance.models import *
import json

# Create your views here.
def addUserView(request):
    username = request.GET.get('name', 'Bob')
    addUser(username)
    return render(request, 'index.html')

def addLocationView(request):
    details = request.POST.copy()
    details['safe'] = "Yes"
    details['people'] = "None"
    addLocation(details)
    return mainPageView(request)

def addLocationHtmlView(request):
    return render(request, 'addLocation.html')

def mainPageView(request):
    locations = getLocations().copy()
    locations = modifyLocations(locations, threshold=20)
    return render(request, 'index.html', {"locations": locations, "num_people":5, "square_footage":100})

def setNumberofPeopleView(request):
    if request.method == 'GET':
        details = request.GET.copy()
    else:
        details = request.POST.copy()
    setNumberofPeople(details)

def setPreferencesView(request):
    copy = request.POST.copy()
    if copy['num_people'] == "": copy['num_people'] = 5
    if copy['square_footage'] == "": copy['square_footage'] = 100
    locations = modifyLocations(getLocations(), threshold=int(copy['num_people']) / int(copy['square_footage']))
    return render(request, 'index.html', {"locations": locations, "num_people":copy['num_people'], "square_footage":copy['square_footage']})
    


