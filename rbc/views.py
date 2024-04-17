from django.http import HttpResponse
from . import utils
from . import forms
import os
from pathlib import Path
from django.http import StreamingHttpResponse
from wsgiref.util import FileWrapper
import mimetypes
import os
from background_task import background
import shutil
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import uuid
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from rest_framework.parsers import JSONParser
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse

# from .consumers import trainingProgressConsumer


class take_input(APIView):
    def post(self, request):
        form = forms.Folder_name_form(request.data)
        if form.is_valid():
            project_name = form.cleaned_data["project_name"]
            folder_name = str(uuid.uuid4())
            os.mkdir(folder_name)
            data_dir = Path(folder_name)
            utils.create_dirs(data_dir)

            return Response(
                {
                    "message": "Input received successfully",
                    "data_dir": str(data_dir),
                    "folder_name": folder_name,
                },
                status=status.HTTP_202_ACCEPTED,
            )

        else:
            return Response(forms.errors, status=status.HTTP_400_BAD_REQUEST)


class data_collection(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        data_dir = request.data.get("data_dir")
        images_to_capture = request.data.get("images_to_capture")
        object_1_label = request.data.get("object_1_label")
        object_2_label = request.data.get("object_2_label")

        frames1 = []
        frames2 = []
        for key, value in request.data.items():
            if key.startswith("frames1"):
                frames1.append(value)
            elif key.startswith("frames2"):
                frames2.append(value)

        utils.video_to_frame(data_dir, object_1_label, frames1)
        utils.train_to_test(data_dir, object_1_label, images_to_capture)
        utils.video_to_frame(data_dir, object_2_label, frames2)
        utils.train_to_test(data_dir, object_2_label, images_to_capture)

        return Response({"message": "Images received"})


class training(APIView):
    def post(self, request):
        data_dir = request.data.get("data_dir")
        folder_name = request.data.get("folder_name")
        data_augmentation = request.data.get("data_augmentation")
        no_of_epochs = request.data.get("no_of_epochs")
        val_accuracy = utils.data_training(
            data_augmentation, data_dir, folder_name, no_of_epochs
        )
        folder_path = os.path.abspath(folder_name)
        delete_folder(folder_path, schedule=600)

        return Response(
            {
                "message": "Model Trained Successfully",
                "accuracy": val_accuracy,
            },
            status=status.HTTP_200_OK,
        )


# class testing(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request):

#         folder_name = request.data.get("folder_name")
#         print(folder_name)
#         if not folder_name:
#             return JsonResponse({"error": "Session data not found."}, status=400)

#         frame = request.data.get("frame")
#         if not frame:
#             return JsonResponse({"error": "No frame received."}, status=400)

#         # Process the received frame and get predictions
#         label, prob1, prob2, same = utils.model_testing(folder_name, frame)
#         print("label", label)
#         print("prob1", prob1)
#         print("prob2", prob2)
#         return Response(
#             {
#                 "label": label,
#                 "probability1": prob1,
#                 "probability2": prob2,
#                 "same": same,
#             },
#             status=status.HTTP_200_OK,
#         )
from .utils import (
    model_testing,
    last_frame,
)  # Import the model_testing function and last_frame variable


class testing(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        object_1_label = request.data.get("object_1_label")
        object_2_label = request.data.get("object_2_label")
        folder_name = request.data.get("folder_name")
        print(folder_name)
        if not folder_name:
            return JsonResponse({"error": "Session data not found."}, status=400)

        frame = request.data.get("frame")
        if not frame:
            return JsonResponse({"error": "No frame received."}, status=400)

        # Process the received frame and get predictions
        label, prob1, prob2, same = model_testing(
            folder_name, frame, object_1_label, object_2_label
        )
        print("label", label)
        print("prob1", prob1)
        print("prob2", prob2)
        return Response(
            {
                "label": label,
                "probability1": prob1,
                "probability2": prob2,
                "same": same,
            },
            status=status.HTTP_200_OK,
        )


# class testing(APIView):
#     def get(self, request):
#         folder_name = request.session.get("folder_name")
#         if not folder_name:
#             return Response(
#                 {"message": "Session data not found."},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         return StreamingHttpResponse(
#             utils.test1(folder_name),
#             content_type="multipart/x-mixed-replace;boundary=frame",
#         )


@background(schedule=300)
def delete_folder(folder_path):
    print(f"Task: Deleting folder '{folder_path}'")
    if folder_path:
        shutil.rmtree(folder_path)


class download_model(APIView):
    def post(self, request):
        folder_name = request.data.get("folder_name")
        print(folder_name)
        if not folder_name:
            return Response(
                {"message": "Session data not found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        model_filename = f"{folder_name}_model.h5"
        model_path = os.path.join(folder_name, model_filename)
        if os.path.exists(model_path):
            content_type, encoding = mimetypes.guess_type(model_path)
            content_type = content_type or "application/octet-stream"
            model_file = open(model_path, "rb")
            response = HttpResponse(FileWrapper(model_file), content_type=content_type)
            response["Content-Disposition"] = f'attachment; filename="{model_filename}"'
            model_file.close()
            return response
        return Response(
            {"message": "Model not found."}, status=status.HTTP_404_NOT_FOUND
        )
