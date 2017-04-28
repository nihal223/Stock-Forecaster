$ ->
  data = []
  if document.getElementById('portfolio-time-val-table') != null
    rows = document.getElementById('portfolio-time-val-table')
    for row in [0 .. rows.getElementsByClassName('portfolio-table-value').length - 1]
      graph_point = []
      date = rows.getElementsByClassName('portfolio-table-date')[row].innerText.trim().split(/-/gm)
      graph_point.push Date.UTC(parseFloat(date[0]), parseFloat(date[1]) - 1, parseFloat(date[2]))
      graph_point.push parseFloat(rows.getElementsByClassName('portfolio-table-value')[row].innerText.trim().substr(1))
      data.push graph_point
    data.reverse()
    $('#portfolio-time-val-graph').highcharts 'StockChart',
      rangeSelector:
        
        inputEnabled: $('#plot').width() > 480
        
      title: text: 'Portfolio Value vs Time'
      series: [ {
        name: 'Portfolio Value'
        data: data
        tooltip: valueDecimals: 2
      } ]
      xAxis:
        tickInterval: 24 * 3600 * 1000 * 3
        minRange: 24 * 3600 * 1000 * 3
    return
  if document.getElementById('generalStockGraph') != null
    data1 =[]
    rows = document.getElementById('stock-portfolio-time-val-table')
    for row in [0 .. rows.getElementsByClassName('stock-portfolio-table-value').length - 1]
      graph_point = []
      date = rows.getElementsByClassName('stock-portfolio-table-date')[row].innerText.trim().split(/-/gm)
      graph_point.push Date.UTC(parseFloat(date[0]), parseFloat(date[1]) - 1, parseFloat(date[2]))
      graph_point.push parseFloat(rows.getElementsByClassName('stock-portfolio-table-value')[row].innerText.trim().substr(1))
      data1.push graph_point
    data1.reverse()
    $('#generalStockGraph').highcharts 'StockChart',
      rangeSelector:
        inputEnabled: $('#plot').width() > 480
      title: text: 'Stock Value vs Time'
      series: [ {
        name: 'Stock Value'
        data: data1
        tooltip: valueDecimals: 2
      } ]
      xAxis:
        tickInterval: 24 * 3600 * 1000 * 3
        minRange: 24 * 3600 * 1000 * 3
    return
return