import sqlite3
import numpy
import math, random, string
import urllib2, time
from datetime import datetime, timedelta
from time import mktime
from yahoo_finance import Share
from scipy.stats import norm
from sklearn.svm import SVR, SVC, LinearSVC

def gethistorical(name):
    try:
        conn = sqlite3.connect('db.sqlite3')
        cur = conn.cursor()
        sql = '''SELECT strftime('%d',Time),strftime('%m',Time),strftime('%Y',Time),Open,Close FROM historical_data_10stocks WHERE symbol LIKE "%''' + name + '''%" LIMIT 240'''
        cur.execute(sql)
        result=cur.fetchall()
        output=[]
        for i in range(0,len(result)):
            temp=[]
            tempstr=str(result[i][0])+'/'+str(result[i][1])+'/'+str(result[i][2])
            temp.append(tempstr)
            temp.append(float(result[i][3]))
            temp.append(float(result[i][4]))
            output.append(temp)
        return output

    except:
    	return 0

def getreal(name):
    try:
        conn = sqlite3.connect('db.sqlite3')
        cur = conn.cursor()
        sql = '''SELECT strftime('%d',Time),strftime('%m',Time),strftime('%Y',Time),Open,Close FROM real2 WHERE symbol LIKE "%''' + name + '''%" LIMIT 240'''
        cur.execute(sql)
        result=cur.fetchall()
        output=[]
        for i in range(0,len(result)):
            temp=[]
            tempstr=str(result[i][0])+'/'+str(result[i][1])+'/'+str(result[i][2])
            temp.append(tempstr)
            temp.append(float(result[i][3]))
            temp.append(float(result[i][4]))
            output.append(temp)
        return output

    except:
        return 0
## ================================================================

#                   Support Vector Machines

## ================================================================

def svm(name, day):
#    Input: Name of the stock, How many days of after current day to get predicted price
#    Output: Predicted Price for next n days

    data = gethistorical(name)
    data = data[::-1]
    open_price_list = []
    close_price_list = []
    predicted_price=[]
    for i in xrange(len(data)):
        open_price_list.append(data[i][1])
        close_price_list.append(data[i][2])
    for iterations in range(day):
        close_price_dataset=[]
        open_price_dataset=[]
        previous_ten_day_close_price_dataset=[]
        g=0
        h=50
        while h<len(close_price_list):
            previous_ten_day_close_price_dataset.append(close_price_list[g:h])
            open_price_dataset.append(open_price_list[h])
            close_price_dataset.append(close_price_list[h])
            g += 1
            h += 1
        moving_average_dataset=[]
        for x in previous_ten_day_close_price_dataset:
            i=0
            for y in x:
                i=i+y
            moving_average_dataset.append(i/10)
        feature_dataset = []
        for j in range(len(close_price_dataset)):
            list = []
            list.append(moving_average_dataset[j])
            list.append(open_price_dataset[j])
            feature_dataset.append(list)
        feature_dataset = numpy.array(feature_dataset)        
        close_price_dataset = numpy.array(close_price_dataset)
        clf = SVR(kernel='rbf',degree=1)
        #print feature_dataset
        #print close_price_dataset
        clf.fit(feature_dataset[-365:],close_price_dataset[-365:])
        target = []
        if iterations==0:
            url_string = "http://www.google.com/finance/getprices?q={0}".format(name)
            stock_info = Share(name)
            list = []
            list.append(stock_info.get_open())
            list.append(stock_info.get_50day_moving_avg())
            target.append(list)
            
        else:
            list = []
            list.append(moving_average_dataset[-1])
            list.append(open_price_dataset[-1])
            target.append(list)

        predicted_close_price = clf.predict(target)[0]
        predicted_price.append(predicted_close_price)
        open_price_list.append(close_price_list[-1])
        close_price_list.append(predicted_close_price)
    
    return predicted_price[29]

## ================================================================

#                   Bayesian Classification

## ================================================================

def bayesian(name):
    data1=gethistorical(name.lower())
    data=[]
    a=60-len(data)
    for i in range(0,len(data1)):
        data.append(data1[i][1])
    #print data 
    x_10 =[]
    for b in xrange(0,a):
        t_data = []
        for i in xrange(len(data) - 10, len(data)):
            t_data.append(data[i])
        for i in xrange(1, 11):
            x_10.append(i)
        t=[]
        t.append(t_data)
        t_data = t
        #print t_data
        N = 10
        M = 6

        rel_err_dr=0

        x=x_10[len(x_10) - 1] + 1

        for k in range(1):
            t = numpy.zeros((N,1),float)
            phi = numpy.zeros((M,1),float)
            phi_sum = numpy.zeros((M,1),float)
            phi_sum_t = numpy.zeros((M,1),float)

            for i in range(M):
                phi[i][0]=math.pow(x,i)

            for i in range(N):
               t[i][0]=t_data[k][i]
                
            for j in range(N):
                for i in range(M):
                    phi_sum[i][0]=phi_sum[i][0]+math.pow(x_10[j],i)
                    phi_sum_t[i][0]=phi_sum_t[i][0]+t[j][0]*math.pow(x_10[j],i)

        # Calculation of variance / standard deviation
            S=numpy.linalg.inv(0.005*numpy.identity(M)+11.1*numpy.dot(phi_sum,phi.T))

            var=numpy.dot((phi.T),numpy.dot(S,phi))
            var=var+1/11.1

        # Calculating the mean
            mean=11.1*numpy.dot(phi.T,numpy.dot(S,phi_sum_t))
           #error_n=0
            #error_n=error_n+math.fabs(t_actual[k]-mean)

            #abs_error=0
            #abs_error = abs_error + error_n
            mean = mean[0][0]
            #print 'mean', mean
            data.append(mean)

    t = t_data[0]
    t_data = t
    sum = 0
    avg = 0
    for i in t_data:
        sum += i
    mov = sum / len(t_data)
    #print 'mov', mov
    per = ((mean - mov) / mov) * 100
    #print 'per', per
    final = []
    mean = round(mean, 3)
    per = round(per, 3)
    final.append(mean)
    final.append(per)
    return final[0]

## ================================================================

#                       NEURAL NETWORKS

## ================================================================

def normalizePrice(price, minimum, maximum):
    return ((2*price - (maximum + minimum)) / (maximum - minimum))

def denormalizePrice(price, minimum, maximum):
    return (((price*(maximum-minimum))/2) + (maximum + minimum))/2


def rollingWindow(seq, windowSize):
    it = iter(seq)
    win = [it.next() for cnt in xrange(windowSize)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win

def getMovingAverage(values, windowSize):
    movingAverages = []
    
    for w in rollingWindow(values, windowSize):
        movingAverages.append(sum(w)/len(w))

    return movingAverages

def getMinimums(values, windowSize):
    minimums = []

    for w in rollingWindow(values, windowSize):
        minimums.append(min(w))
            
    return minimums

def getMaximums(values, windowSize):
    maximums = []

    for w in rollingWindow(values, windowSize):
        maximums.append(max(w))

    return maximums


def getTimeSeriesValues(values, window):
    movingAverages = getMovingAverage(values, window)
    minimums = getMinimums(values, window)
    maximums = getMaximums(values, window)

    returnData = []

    # build items of the form [[average, minimum, maximum], normalized price]
    for i in range(0, len(movingAverages)):
        inputNode = [movingAverages[i], minimums[i], maximums[i]]
        price = normalizePrice(values[len(movingAverages) - (i + 1)], minimums[i], maximums[i])
        outputNode = [price]
        tempItem = [inputNode, outputNode]
        returnData.append(tempItem)

    return returnData


def getHistoricalData(stockSymbol):
    historicalPrices = []
    
    # login to API
    urllib2.urlopen("http://api.kibot.com/?action=login&user=guest&password=guest")

    # get 14 days of data from API (business days only, could be < 10)
    url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=14&unadjusted=1&regularsession=1"
    apiData = urllib2.urlopen(url).read().split("\n")
    for line in apiData:
        if(len(line) > 0):
            tempLine = line.split(',')
            price = float(tempLine[1])
            historicalPrices.append(price)

    return historicalPrices


def getTrainingData(stockSymbol):
    historicalData = getHistoricalData(stockSymbol)

    # reverse it so we're using the most recent data first, ensure we only have 9 data points
    historicalData.reverse()
    del historicalData[9:]

    # get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
    trainingData = getTimeSeriesValues(historicalData, 5)

    return trainingData

def getPredictionData(stockSymbol):
    historicalData = getHistoricalData(stockSymbol)

    # reverse it so we're using the most recent data first, then ensure we only have 5 data points
    historicalData.reverse()
    del historicalData[5:]

    # get five 5-day moving averages, 5-day lows, and 5-day highs
    predictionData = getTimeSeriesValues(historicalData, 5)

    # remove associated closing price
    predictionData = predictionData[0][0]

    return predictionData


def analyzeSymbol(stockSymbol):
    startTime = time.time()
    
    trainingData = getTrainingData(stockSymbol)
    
    network = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 1)

    network.train(trainingData)

    # get rolling data for most recent day
    predictionData = getPredictionData(stockSymbol)

    # get prediction
    returnPrice = network.test(predictionData)

    # de-normalize and return predicted stock price
    predictedStockPrice = denormalizePrice(returnPrice, predictionData[1], predictionData[2])

    # create return object, including the amount of time used to predict
    returnData = {}
    returnData['price'] = predictedStockPrice
    returnData['time'] = time.time() - startTime

    return returnData['price']

## ================================================================

# calculate a random number a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill = 0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    # tanh is a little nicer than the standard 1/(1+e^-x)
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        # number of input, hidden, and output nodes
        self.inputNodes = inputNodes + 1 # +1 for bias node
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # activations for nodes
        self.inputActivation = [1.0]*self.inputNodes
        self.hiddenActivation = [1.0]*self.hiddenNodes
        self.outputActivation = [1.0]*self.outputNodes
        
        # create weights
        self.inputWeight = makeMatrix(self.inputNodes, self.hiddenNodes)
        self.outputWeight = makeMatrix(self.hiddenNodes, self.outputNodes)
        # set them to random vaules
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes):
                self.inputWeight[i][j] = rand(-0.2, 0.2)
        for j in range(self.hiddenNodes):
            for k in range(self.outputNodes):
                self.outputWeight[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.inputNodes, self.hiddenNodes)
        self.co = makeMatrix(self.hiddenNodes, self.outputNodes)

    def update(self, inputs):
        if len(inputs) != self.inputNodes-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.inputNodes-1):
            self.inputActivation[i] = inputs[i]

        # hidden activations
        for j in range(self.hiddenNodes):
            sum = 0.0
            for i in range(self.inputNodes):
                sum = sum + self.inputActivation[i] * self.inputWeight[i][j]
            self.hiddenActivation[j] = sigmoid(sum)

        # output activations
        for k in range(self.outputNodes):
            sum = 0.0
            for j in range(self.hiddenNodes):
                sum = sum + self.hiddenActivation[j] * self.outputWeight[j][k]
            self.outputActivation[k] = sigmoid(sum)

        return self.outputActivation[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.outputNodes:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.outputNodes
        for k in range(self.outputNodes):
            error = targets[k]-self.outputActivation[k]
            output_deltas[k] = dsigmoid(self.outputActivation[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.hiddenNodes
        for j in range(self.hiddenNodes):
            error = 0.0
            for k in range(self.outputNodes):
                error = error + output_deltas[k]*self.outputWeight[j][k]
            hidden_deltas[j] = dsigmoid(self.hiddenActivation[j]) * error

        # update output weights
        for j in range(self.hiddenNodes):
            for k in range(self.outputNodes):
                change = output_deltas[k]*self.hiddenActivation[j]
                self.outputWeight[j][k] = self.outputWeight[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes):
                change = hidden_deltas[j]*self.inputActivation[i]
                self.inputWeight[i][j] = self.inputWeight[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.outputActivation[k])**2
            
        return error


    def test(self, inputNodes):
        # print(inputNodes, '->', self.update(inputNodes))
        return self.update(inputNodes)[0]

    def weights(self):
        print('Input weights:')
        for i in range(self.inputNodes):
            print(self.inputWeight[i])
        print()
        print('Output weights:')
        for j in range(self.hiddenNodes):
            print(self.outputWeight[j])

    def train(self, patterns, iterations = 1000, N = 0.5, M = 0.1):
        # N: learning rate, M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
        







stocklist = ['GOOG', 'YHOO', 'AMZN', 'AAPL', 'FB', 'INTL', 'ORCL', 'TSLA', 'GE', 'CSCO']

conn = sqlite3.connect('db.sqlite3')
cur = conn.cursor()
cur.execute('''DROP TABLE stockapp_prediction''')
cur.execute('''CREATE TABLE stockapp_prediction
   (ID INTEGER PRIMARY KEY AUTOINCREMENT,
   Symbol VARCHAR,
   svm_prediction REAL,
   bayesian_prediction REAL,
   ANN_prediction REAL
    );''')

for stock in stocklist:
    print stock
    print "----------"
    s=svm(stock,30)
    print "SVM Prediction: "+str(s)
    B= bayesian(stock)
    print "Bayesian Prediction: "+str(B)
    ann=analyzeSymbol(stock)
    print "Neural Network Prediction: "+str(ann)
    cur.execute ("""
	            INSERT INTO stockapp_prediction (symbol, svm_prediction, bayesian_prediction, ANN_prediction)
	            VALUES
	                (?, ?, ?, ?) 
	       """, (stock, s, B, ann))
    
conn.commit() 
cur.close() 

