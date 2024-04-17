from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("live_sketch", views.live_sketch, name="live_sketch"),
    path("generate2", views.generate2, name="generate2"),
    path("download_photo", views.download_photo, name="download_photo"),
]
