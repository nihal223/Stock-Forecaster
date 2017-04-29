'''views.py to support views for the porfolio of each user'''
from django.shortcuts import render
# Takes care of checking if the user is logged in or not
from django.contrib.auth.decorators import login_required
# Yahoo YQL stockretiever to get the stock infos
from lib.yahoo_stock_scraper.stockretriever import get_current_info, get_historical_info, get_3_month_info, get_news_feed, get_12_month_info
# For workbook to create the historical data of a stock
import xlwt
# Http clients to send the attachment file for historical data
from django.http import HttpResponse
# models i.e. database
from .models import StockPortfolio, StockFolioUser
# Get the global stock ticker symbol
from . import STOCK_SYMBOLS
# Need simplejson to pass dictionart STOCK_SYMBOLS to templates
import json
import sqlite3

import datetime
import random
import django
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from django import forms
from yahoo_finance import Share

@login_required
def portfolio(request):
  '''The main method for all the user functionality'''
  user_id = request.user.id
  if request.method == 'POST':
    which_form = request.POST.get('which-form', '').strip()

    if which_form == 'find-stock':
      symbol = request.POST.get('stock', '').strip().split(' ')[0].strip().upper()
      if symbol != '':
        portfolio_stock = portfolio_stocks(user_id)
        money = portfolio_stock['money']
        porfolio = portfolio_stock['portfolio_info']



        return render(request, 'StockFolio/portfolio.html', {'stock':get_current_info([''+symbol]), 'news' : get_news_feed(symbol), 'portfolio' : porfolio, 'stock_rows'  : plot(user_id), 'symbols' : json.dumps(STOCK_SYMBOLS), 'money' : money})

    elif which_form == 'download-historical':
      return download_historical(request.POST.get('stock-symbol', '').strip())

    elif which_form == 'buy-stock':
      symbol = request.POST.get('stock-symbol', '').strip()
      StockPortfolio.buy(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      portfolio_stock = portfolio_stocks(user_id)
      money = portfolio_stock['money']
      porfolio = portfolio_stock['portfolio_info']
      return render(request, 'StockFolio/portfolio.html', {'stock':get_current_info([''+symbol]), 'news' : get_news_feed(symbol), 'portfolio' : porfolio, 'portfolio_rows' : plot(user_id), 'symbols' : json.dumps(STOCK_SYMBOLS), 'money' : money})

    elif which_form =='recommend':
      symbol = request.POST.get('stock-symbol', '').strip()
      portfolio_stock = portfolio_stocks(user_id)
      money = portfolio_stock['money']
      porfolio = portfolio_stock['portfolio_info']
      reco_svm = []
      my_stocks = []
      conn = sqlite3.connect('PyFolio.sqlite3')
      cur = conn.cursor()
      sql = ''' SELECT svm_prediction FROM stockapp_prediction_svm_all WHERE symbol LIKE "%''' + symbol + '''%" ''' 
      sql2 = ''' SELECT ANN_prediction FROM stockapp_prediction_svm_all WHERE symbol LIKE "%''' + symbol + '''%" ''' 
      sql3 = ''' SELECT bayesian_prediction FROM stockapp_prediction_svm_all WHERE symbol LIKE "%''' + symbol + '''%" '''
      
      cur.execute(sql)
      result=cur.fetchall()

      cur.execute(sql2)
      reco_ann=cur.fetchall()

      cur.execute(sql3) 
      reco_bay = cur.fetchall()


      for i in range(30):
        reco_svm.append(result[i][0])

      stock_price = float(Share(symbol).get_price())
      for i in range(len(portfolio_stock['portfolio_info'])):
        my_stocks.append(portfolio_stock['portfolio_info'][i]['symbol'])

      x,y=0,0

      if float(reco_ann[0][0])>float(stock_price)+5 or float(reco_svm[29])>1.05*float(stock_price):
        x,y=0,0
        if float(reco_ann[0][0])>float(stock_price)+5 and float(reco_ann[0][0])<=1.1*float(stock_price):
          x=1
        elif float(reco_svm[29])>1.05*float(stock_price) and float(reco_svm[29])<=1.1*float(stock_price):
          y=1

        if float(reco_ann[0][0])>float(stock_price)+5 and float(reco_svm[29])>1.05*float(stock_price):
          x,y=1,1

        if (x==1 and y==0):
          message="Buy for Short term gains"

        elif (y==1 and x==0):
          message="Buy for Long term gains"

        elif (y==1 and x==1):
          message="Buy for short and long term gains"

      if symbol in my_stocks:    
        if float(reco_ann[0][0])<float(stock_price)+5 and float(reco_ann[0][0])>float(stock_price)-5 and float(reco_svm[29])<=1.05*float(stock_price) and float(reco_svm[29])>0.95*float(stock_price):
          x,y=0,0
          if float(reco_ann[0][0])<float(stock_price)+5 and float(reco_ann[0][0])>float(stock_price)-5:
            x=1
          elif float(reco_svm[29])<=1.05*float(stock_price) and float(reco_svm[29])>0.95*float(stock_price):
            y=1

          if float(reco_ann[0][0])<float(stock_price)+5 and float(reco_ann[0][0])>float(stock_price)-5 and float(reco_svm[29])<=1.05*float(stock_price) and float(reco_svm[29])>0.95*float(stock_price):
            x,y=1,1

          if (x==1 and y==0):
            message="HOLD"

          elif (y==1 and x==0):
            message="HOLD"

          elif (y==1 and x==1):
            message="HOLD"
      else:
        message = "Stock not in your portfolio"

      if symbol in my_stocks:
        if float(reco_ann[0][0])<float(stock_price)-5 and float(reco_ann[0][0])>=0.90*float(stock_price) or float(reco_svm[29])<0.95*float(stock_price) and float(reco_svm[29])>0.90*float(stock_price):
          x,y=0,0
          if float(reco_ann[0][0])>float(stock_price)-5 and float(reco_ann[0][0])>0.90*float(stock_price):
            x=1
          elif float(reco_svm[29])>0.95*float(stock_price) and float(reco_svm[29])>0.90*float(stock_price):
            y=1

          if float(reco_ann[0][0])>float(stock_price)-5 and float(reco_ann[0][0])>=0.90*float(stock_price) and float(reco_svm[29])>0.95*float(stock_price) and float(reco_svm[29])>0.90*float(stock_price):
            x,y=1,1

          if (x==1 and y==0):
            message="SELL"

          elif (y==1 and x==0):
            message="SELL"

          elif (y==1 and x==1):
            message="SELL"
      else:
        message = "Stock not in your portfolio"

      ann_pred = "Tomorrow's predicted price: " + str(reco_ann[0][0])


      return render(request, 'StockFolio/portfolio.html', {'stock':get_current_info([''+symbol]), 'news' : get_news_feed(symbol), 'portfolio' : porfolio, 'portfolio_rows' : plot(user_id), 'symbols' : json.dumps(STOCK_SYMBOLS), 'money' : money, 'value_reco': message, 'ann_pred': ann_pred})

    elif which_form == 'buy-sell':
      symbol = request.POST.get('stock-symbol', '').strip()
      if request.POST.get('buy-stock'):
        StockPortfolio.buy(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      elif request.POST.get('sell-stock'):
        StockPortfolio.sell(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      portfolio_stock = portfolio_stocks(user_id)
      money = portfolio_stock['money']
      porfolio = portfolio_stock['portfolio_info']
      return render(request, 'StockFolio/portfolio.html', {'stock':get_current_info([''+symbol]), 'news' : get_news_feed(symbol), 'portfolio' : porfolio, 'portfolio_rows' : plot(user_id), 'symbols' : json.dumps(STOCK_SYMBOLS), 'money' : money})

  portfolio_stock = portfolio_stocks(user_id)
  money = portfolio_stock['money']
  porfolio = portfolio_stock['portfolio_info']

  return render(request, 'StockFolio/portfolio.html', {'portfolio' : porfolio, 'portfolio_rows' : plot(user_id), 'symbols' : json.dumps(STOCK_SYMBOLS), 'money' : money})

def download_historical(symbol):
  '''Downloads the historical data to the desktop'''
  stock_history = get_historical_info(symbol)
  book = xlwt.Workbook()
  sheet = book.add_sheet('Sheet')
  for idx, key in enumerate(stock_history[0].keys()):
    sheet.write(0, idx, key)
  for idx, row in enumerate(stock_history, start=1):
    for col, val in enumerate(row.values()):
      sheet.write(idx, col, val)
  response = HttpResponse(content_type='application/x-msexcel')
  response['Percentagma'] = 'no-cache'
  response['Content-disposition'] = 'attachment; filename=' + symbol + '-history.xls'
  book.save(response)
  return response

def portfolio_stocks(user_id):
  '''Returns the list of stocks in a users portfolio'''
  portfolio_info = []
  stock_list = StockPortfolio.objects.filter(user=user_id)
  user = StockFolioUser.objects.filter(user=user_id)[0]
  money = {'spent' : user.spent, 'earnt' : user.earnt, 'value' : 0, 'profit': '+'}
  if stock_list:
    symbols = [stock.stock for stock in stock_list]
    if len(symbols) == 1:
      stock_data = [get_current_info(symbols)]
    else:
      stock_data = get_current_info(symbols)
    for stock in stock_data:
      for stock_from_list in stock_list:
        if stock_from_list.stock == stock['Symbol']:
          stock['shares'] = stock_from_list.shares
          stock['cost'] = int(stock_from_list.shares) * float(stock['LastTradePriceOnly'])
          money['value'] += float(stock['cost'])
    portfolio_info = [stock_data] if stock_data.__class__ == dict else stock_data
  if float(money['spent']) > (float(money['value']) + float(money['earnt'])):
    money['profit'] = '-'
  return {'portfolio_info' : portfolio_info, 'money' : money}

def plot(user_id):
  '''Gets Months of historical info on stock and for the graph plots of portfolio'''
  rows = []
  stocks = StockPortfolio.objects.filter(user=user_id)
  if stocks:
    data = [list(reversed(get_12_month_info(stock.stock))) for stock in stocks]
    days = [day['Date'] for day in data[0]]
    for idx, day in enumerate(days):
      first = True
      for stock_index, stock in enumerate(data):
        if len(stock) <= idx:
          continue
        if first:
          row = {'Value' : round(float(stock[idx]['Close']) * StockPortfolio.objects.filter(stock=stocks[stock_index].stock, user=user_id)[0].shares, 2), 'Date' : day, 'Percent': (float(stock[idx]['Open']) - float(stock[idx]['Close'])) / float(stock[idx]['Close']) * 100, 'Volume': int(stock[idx]['Volume']), 'High': float(stock[idx]['High']), 'Low': float(stock[idx]['Low']), 'AdjClose': float(stock[idx]['AdjClose']), 'Open': float(stock[idx]['Open'])}
          first = False
        else:
          row['Date'] = day
          row['Value'] += round(float(stock[idx]['Close']) * StockPortfolio.objects.filter(stock=stocks[stock_index].stock, user=user_id)[0].shares, 2)
          row['Volume'] += int(stock[idx]['Volume'])
          row['Open'] = (row['Open']  + float(stock[idx]['Open']))/2
          row['High'] = (row['High']  + float(stock[idx]['High']))/2
          row['Low'] = (row['Low']  + float(stock[idx]['Low']))/2
          row['AdjClose'] = (row['AdjClose']  + float(stock[idx]['AdjClose']))/2
          row['Percent'] += (float(stock[idx]['Open']) - float(stock[idx]['Close'])) / float(stock[idx]['Close']) * 100
      rows.append(row)
    rows.reverse()
  return rows
    
def plotStock(stock_id):
  '''Gets Months of historical info on stock and for the graph plots of portfolio'''
  rows = []
  stocks = stock_id
  if stocks:
    data = [list(reversed(get_12_month_info(stocks)))]
    days = [day['Date'] for day in data[0]]
    for idx, day in enumerate(days):
      first = True
      for stock_index, stock in enumerate(data):
        if len(stock) <= idx:
          continue
        if first:
          row = {'Value' : round(float(stock[idx]['Close']), 2), 'Date' : day, 'Percent': (float(stock[idx]['Open']) - float(stock[idx]['Close'])) / float(stock[idx]['Close']) * 100, 'Volume': int(stock[idx]['Volume']), 'High': float(stock[idx]['High']), 'Low': float(stock[idx]['Low']), 'AdjClose': float(stock[idx]['AdjClose']), 'Open': float(stock[idx]['Open'])}
          first = False
        else:
          row['Date'] = day
          row['Value'] += round(float(stock[idx]['Close']), 2)
          row['Volume'] += int(stock[idx]['Volume'])
          row['Open'] = (row['Open']  + float(stock[idx]['Open']))/2
          row['High'] = (row['High']  + float(stock[idx]['High']))/2
          row['Low'] = (row['Low']  + float(stock[idx]['Low']))/2
          row['AdjClose'] = (row['AdjClose']  + float(stock[idx]['AdjClose']))/2
          row['Percent'] += (float(stock[idx]['Open']) - float(stock[idx]['Close'])) / float(stock[idx]['Close']) * 100
      rows.append(row)
    rows.reverse()
  #return rows

def svm_plot(request):
  if request.method == 'POST':
    which_form = request.POST.get('which-form', '').strip()
    if which_form == 'svm-plot':
      symbol = request.POST.get('stock-symbol', '').strip()
  conn = sqlite3.connect('PyFolio.sqlite3')
  cur = conn.cursor()
  sql = ''' SELECT svm_prediction FROM stockapp_prediction_svm_all WHERE symbol LIKE "%''' + symbol + '''%" ''' 
  cur.execute(sql)
  result=cur.fetchall()
  fig=Figure()
  ax=fig.add_subplot(111)
  x=[]
  y=[]
  now=datetime.datetime.now()
  delta=datetime.timedelta(days=1)
  for i in range(30):
      x.append(now)
      now+=delta
      y.append(result[i][0])
  ax.plot_date(x, y, '-')
  ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
  fig.autofmt_xdate()
  canvas=FigureCanvas(fig)
  response=django.http.HttpResponse(content_type='image/png')
  canvas.print_png(response)
  return response

def obv_plot(request):
  if request.method == 'POST':
    which_form = request.POST.get('which-form', '').strip()
    if which_form == 'obv-plot':
      symbol = request.POST.get('stock-symbol', '').strip()
  conn = sqlite3.connect('PyFolio.sqlite3')
  cur = conn.cursor()
  sql = ''' SELECT Close,Volume FROM historical_data_10stocks WHERE symbol LIKE "%''' + symbol + '''%" ''' 
  cur.execute(sql)
  result=cur.fetchall()
  x=[]
  y=[]
  prev_obv=0
  for i in range(60):
    if result[i][0] < result[i+1][0]:
      y.append((prev_obv+result[i+1][1]))
    elif result[i][0] > result[i+1][0]:
      y.append((prev_obv-result[i+1][1]))
    else:
      y.append(prev_obv)
  y.reverse()
  print(y)    
  fig=Figure()
  ax=fig.add_subplot(111)
  now=datetime.datetime.now()
  delta=datetime.timedelta(days=1)
  for i in range(60):
      x.append(now)
      now+=delta
  ax.plot_date(x, y, '-')
  ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
  fig.autofmt_xdate()
  canvas=FigureCanvas(fig)
  response=django.http.HttpResponse(content_type='image/png')
  canvas.print_png(response)
  return response


