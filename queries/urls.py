#queries/urls.py
from django.conf.urls import url
from queries import views

urlpatterns = [
	url(r'^$', views.AboutPageView.as_view()),
	url(r'^runquery/', views.runquery, name='runquery'),
]
