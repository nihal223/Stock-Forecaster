import sqlite3
from yahoo_finance import Share
#yahoo = Share('GOOG')
company_list = ['GOOG', 'YHOO', 'AMZN', 'AAPL', 'FB', 'INTC', 'IBM', 'MSFT', 'TWTR', 'HPQ']
li=[]
for i in range(len(company_list)):
	comp=Share(company_list[i])
	li.append(comp.get_historical('2016-04-25', '2017-04-26'))

print li[1][1]
#comp=Share(company_list[1])
#li.append(comp.get_historical('2014-04-25', '2015-04-25'))

conn = sqlite3.connect('PyFolio.sqlite3')
cur = conn.cursor()
cur.execute('''DROP TABLE historical_data_10stocks''')
cur.execute('''CREATE TABLE historical_data_10stocks
       (ID INTEGER PRIMARY KEY AUTOINCREMENT,
       Symbol		VARCHAR,
       Time         DATE    NOT NULL,
       Open         REAL     NOT NULL,
       Close        REAL,
       High         REAL,
       Low          REAL,
       Volume 		BIGINT   
       );''')

#print len(li[i])
for i in range(len(company_list)):
	for j in range(len(li[i])):
		cur.execute("INSERT INTO historical_data_10stocks (Symbol,Time,Open,Close,High,Low,Volume) \
			VALUES (?,?,?,?,?,?,?)",(li[i][j]['Symbol'],li[i][j]['Date'],li[i][j]['Open'],li[i][j]['Close'],li[i][j]['High'],li[i][j]['Low'],li[i][j]['Volume']))
#cur.execute('''COPY (SELECT * FROM hist2 WHERE Symbol='YHOO') TO '/home/akshay/software_csv/yhoo_hist.csv' DELIMITER ',' CSV HEADER;''')
#cur.execute('''COPY (SELECT * FROM hist2 WHERE Symbol='AAPL') TO '/home/akshay/software_csv/aapl_hist.csv' DELIMITER ',' CSV HEADER;''')
#cur.execute('''COPY (SELECT * FROM hist2 WHERE Symbol='GOOG') TO '/home/akshay/software_csv/goog_hist.csv' DELIMITER ',' CSV HEADER;''')
#cur.execute('''COPY (SELECT * FROM hist2 WHERE Symbol='FB') TO '/home/akshay/software_csv/fb_hist.csv' DELIMITER ',' CSV HEADER;''')
#cur.execute('''COPY (SELECT * FROM hist2 WHERE Symbol='AMZN') TO '/home/akshay/software_csv/amzn_hist.csv' DELIMITER ',' CSV HEADER;''')
conn.commit()
conn.close()


