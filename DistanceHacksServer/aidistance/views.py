from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from aidistance.models import *
import json

# Create your views here.
def addUserView(request):
    username = request.GET.get('name', 'Bob')
    addUser(username)
    return render(request, 'signin.html')

def addLocationView(request):
    name = request.GET.get('location', 'Empty')
    details = request.GET.copy()


    addLocation(details)
    return render(request, 'signin.html')