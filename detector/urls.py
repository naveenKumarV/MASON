from django.urls import path

from . import views

app_name = 'detector'
urlpatterns = [
    path('', views.index, name='index'),
    path('bboxes', views.get_bboxes, name='bboxes'),
    path('adjust', views.adjust_bbox, name='adjust'),
]