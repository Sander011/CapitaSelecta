from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from modelling import views
from .views import index
router = routers.DefaultRouter()
router.register(r'datasets', views.DatasetViewSet, 'dataset')

urlpatterns = [
    path('', index, name='index'),
    path('admin/', admin.site.urls),
    path('api/', include(router.urls))
]