from django.shortcuts import render
from django.http import StreamingHttpResponse
from .utils import uvbc


def cleek(request):
    return render(request, "vbc/cleek.html")


def star(request):
    return render(request, "vbc/vbc.html")


def live_sketch(request):
    return render(request, "live_sketch/live_sketch.html")


def vbc(request):
    response = StreamingHttpResponse(
        uvbc(),
        content_type="multipart/x-mixed-replace;boundary=frame",
    )
    return response
