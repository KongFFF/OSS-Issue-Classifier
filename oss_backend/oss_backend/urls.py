from django.contrib import admin
from django.urls import path
from classifier import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'), # 原有功能
    path('scan/', views.scan_repo, name='scan'),     # 新增功能
]