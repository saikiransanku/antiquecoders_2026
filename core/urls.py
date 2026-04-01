from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # Custom account routes to support email/phone login and signup
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/signup/', views.signup_view, name='signup'),
    path('accounts/logout/', views.logout_view, name='logout'),
]
