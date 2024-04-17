# your_project_name/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("rbc.urls")),
    path("", include("vbc.urls")),
    path("", include("live_sketch.urls")),
]
