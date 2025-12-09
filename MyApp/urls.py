from __future__ import annotations

from django.urls import path

from . import views

app_name = "MyApp"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_csv_view, name="upload_csv"),
    path("profile/", views.profile_view, name="profile"),
    path("model-analysis/", views.model_analysis_view, name="model_analysis"),
]

