# your_app_name/urls.py

from django.urls import path
from rbc import views

urlpatterns = [
    path("rbc/take_input/", views.take_input.as_view(), name="take_input"),
    path(
        "rbc/data_collection/", views.data_collection.as_view(), name="data_collection"
    ),
    path("rbc/training/", views.training.as_view(), name="training"),
    path("rbc/testing/", views.testing.as_view(), name="testing"),
    path("rbc/download_model/", views.download_model.as_view(), name="download_model"),
]
