from django.http import StreamingHttpResponse
import cv2
import numpy as np
from django.shortcuts import render
from . import utils


def live_sketch(request):
    return render(request, "live_sketch/live_sketch.html")


def generate2(request):
    def generate():
        yield from utils.generate2()

    return StreamingHttpResponse(
        generate(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


from django.http import HttpResponse
import cv2
import os


def download_photo(request):
    def download_photo2():
        yield from utils.download_photo()

    return StreamingHttpResponse(
        download_photo2(), content_type="multipart/x-mixed-replace; boundary=frame"
    )
