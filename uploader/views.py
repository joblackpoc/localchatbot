from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadForm
from .models import UploadedFile
from .utils import extract_text_from_file
from chat.rag import ingest_text  # your RAG ingestion function

def upload_file(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()  # save file

            file_path = settings.MEDIA_ROOT / obj.file.name
            text = extract_text_from_file(str(file_path))

            obj.extracted_text = text
            obj.save()

            # send text to RAG
            ingest_text(text)

            return render(request, "uploader/upload_success.html", {"file": obj})

    else:
        form = UploadForm()

    return render(request, "uploader/upload.html", {"form": form})
