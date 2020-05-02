from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json

# Create your views here.

def signinView(request):
	return render(request, 'signin.html')

def manifestView(request):
	with open('static/json/manifest.json') as f:
		data = json.load(f)
		print(data)
	return HttpResponse(json.dumps(data), content_type="application/json")
