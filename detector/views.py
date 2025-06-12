# detector/views.py
import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

def home(request):
    result = None
    uploaded_image_url = None

    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_image_url = fs.url(filename)

        # Placeholder detection logic:
        from random import choice
        result = choice(['Dog Detected 🐶', 'Cat Detected 🐱'])

    return render(request, 'detector/home.html', {
        'result': result,
        'uploaded_image_url': uploaded_image_url
    })
