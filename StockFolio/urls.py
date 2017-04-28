'''Urls for the StockFolio app'''
from django.conf.urls import url
from . import views as StockFolio_views

urlpatterns = [
	#url(r'^recommend', StockFolio_views.recommend, name="recommend"),
    url(r'^svm_plot', StockFolio_views.svm_plot, name="svm_plot"),
    url(r'^(?i)portfolio', StockFolio_views.portfolio, name="portfolio"),
]
