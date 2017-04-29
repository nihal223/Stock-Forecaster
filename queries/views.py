# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django import forms
from django.views.generic import TemplateView
from django.contrib.auth import login, authenticate
from django.shortcuts import redirect
from django.template import Context
import sqlite3

class AboutPageView(TemplateView):
    template_name = "index.html"

def runquery(request):
	#form = ContactForm(request.POST)
	#if form.is_valid():
	#	form.save()
	companyname = request.POST.get('companyname')
	querynumber = request.POST.get('query')
	c="INVALID"

	#c = Context({"companyname": companyname})
	conn = sqlite3.connect('PyFolio.sqlite3')
	cur = conn.cursor()
	result='none'
	if(querynumber=='1'):
		row=cur.execute('''SELECT DISTINCT Symbol,Price FROM realtime_data ORDER BY time DESC LIMIT 10''')
		result1 = row.fetchall()
		result = []
		for x in result1:
			result.append(x)
	elif(querynumber=='2'):
		row=cur.execute('''SELECT MAX(Close) FROM (SELECT * FROM historical_data_10stocks WHERE symbol=(?) LIMIT 10)''',(companyname,))
		result = row.fetchone()
	elif(querynumber=='3'):
		row=cur.execute('''SELECT AVG(Close) FROM (SELECT * FROM historical_data_10stocks WHERE symbol=(?) LIMIT 365)''',(companyname,))
		result=row.fetchone()
	elif(querynumber=='4'):
		row=cur.execute('''SELECT MIN(Close) FROM (SELECT * FROM historical_data_10stocks WHERE symbol=(?) LIMIT 365)''',(companyname,))
		result = row.fetchone()
	elif(querynumber=='5'):
		row=cur.execute('''SELECT Symbol,AVG(Close) from historical_data_10stocks GROUP BY Symbol HAVING AVG(Close) < (SELECT MIN(Close) FROM historical_data_10stocks WHERE Symbol=(?) GROUP BY Symbol) ''',(companyname,))
		result = row.fetchall()
		# result = list(result)
	
	return render(request, "index.html", {"companyname": result})

	conn.commit()
	conn().close()

    
