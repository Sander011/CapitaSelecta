from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from modelling import views

router = routers.DefaultRouter()
router.register(r'datasets', views.DatasetViewSet, 'dataset')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls))
]