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
    return render(request, 'index.html')

def setNumberofPeopleView(request):
    if request.method == 'GET':
        details = request.GET.copy()
    else:
        details = request.POST.copy()
    setNumberofPeople(details)
