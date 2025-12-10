from django.urls import path
from . import views
from uploader import views_chat

urlpatterns = [
    path("upload/", views.upload_file, name="upload_file"),
    path("chat-upload/", views_chat.chat_upload, name="chat_upload"),
]
