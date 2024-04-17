from django.urls import path
from . import views

urlpatterns = [
    path("vbc/", views.vbc, name="vbc"),
    path("cleek/", views.cleek, name="cleek"),
    path("star/", views.star, name="star"),
]
