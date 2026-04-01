from django.urls import path
from . import views

urlpatterns = [
    path('diagnose/', views.diagnose_plant, name='diagnose_plant'),
]