import numpy
import os
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import threading

def shutdown():
    print('自动休眠啦。。。')
    time.sleep(3)
    os.system('rundll32.exe powrProf.dll,SetSuspendState')
    #os.system('shutdown -s -f -t 1')

def exit():
    inputExit = input('按y退出程序:\n')
    if  str(inputExit) == 'y':
        os._exit(0)

def MAPE(real,prediction):
    mape = []
    avg_real=float(sum(real)/len(real))
    for i in range(len(real)):
        mape.append(abs(float(real[i])-float(prediction[i]))/avg_real)
    return sum(mape)/float(len(real)),mape

def MAPE2(real,prediction):
    mape = []
    #avg_real=float(sum(real)/len(real))
    for i in range(len(real)):
        mape.append(abs(float(real[i])-float(prediction[i]))/abs(float(real[i])))
    return sum(mape)/float(len(mape)),mape

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def data_input(s,shape,look_back,scaler,Gaus_filter=False,pic=True):
    #dataframe = read_csv('D:\pythonproject\Oracle_python\负荷预测\id_data\kalman\id_name_data_50924_8-28_kalman.txt', usecols=[0], engine='python')
    dataframe = read_csv(s, usecols=[0],header=None, engine='python')
    dataset = dataframe.values

    #dataset=dataset[:-100,:]

    # 将整型变为float
    dataset = dataset.astype('float32')
    if Gaus_filter==True:
        dataset = filters.gaussian_filter(dataset,1.7)

    if pic==True:
        plt.plot(dataset)
        plt.show()

    # fix random seed for reproducibility
    numpy.random.seed(7)


    # normalize the dataset
    #scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)


    dataX,dataY=create_dataset(dataset, look_back)
    trainX, trainY=dataX[:-24,:],dataY[:-24]
    preX=dataX[-24:-23,:]

    if shape==0:
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        preX= numpy.reshape(preX, (preX.shape[0], 1, preX.shape[1]))
    elif shape==1:
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
        #testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1],1))
        preX= numpy.reshape(preX, (preX.shape[0], preX.shape[1],1))

    return trainX,trainY,preX,dataset

if __name__ == '__main__':
    auto_shutdown = False
    data_output = True
    shape = 0   #0为将序列视为一个整体，1为将序列视为多个时间步的组合
    look_back = 100
    scaler = MinMaxScaler(feature_range=(0, 1))


    if auto_shutdown==True:
        print('\033[1;31;40m')
        print('注意！程序运行结束将自动关闭电脑！')
        print('\033[0m')


    # create and fit the LSTM network
    model = Sequential()
    if shape==0:
        model.add(LSTM(128, input_shape=(1, look_back),return_sequences=True))
    elif shape==1:
        model.add(LSTM(128, input_shape=(look_back,1),return_sequences=True))
    # model.add(Dropout(0.1))
    #model.add(LSTM(32,return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(96,return_sequences=False))
    # model.add(Dropout(0.1))
    #model.add(Dense(32))
    # model.add(Dropout(0.1))
    model.add(Dense(16))
    #model.add(LSTM(16,return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


    # files = os.listdir('./data_9_20/1/xiaobo')
    # print(files)
    # for file in files:
    #     print('------------------------%s-----------------------'%file)
    #     trainX, trainY, testX, testY, preX, dataset = data_input('data_9_20/1/xiaobo/%s'%file,look_back,scaler,Gaus_filter=False,pic=False)
    #     model.fit(trainX, trainY, epochs=140, batch_size=672, verbose=1,shuffle=False)


    # #trainX, trainY, testX, testY, preX, dataset = data_input('data_9_20/1/661810.txt',look_back,scaler,Gaus_filter=True)
    trainX, trainY, preX, dataset = data_input('price(09).txt',shape,look_back,scaler,Gaus_filter=False,pic=True)
    print('dataset:',dataset.shape)
    print('trainX:',trainX.shape)
    print('preX:',preX.shape)
    print('aaaaaaaaa', preX.size)

    history=model.fit(trainX, trainY, epochs=5000, batch_size=256, verbose=1, shuffle=False)#,validation_split=0.1)


    # make predictions
    trainPredict = model.predict(trainX)
    #print(trainPredict.shape)
    ######################################################################
    preY=[]
    for i in range(24):
        prePredict=model.predict(preX)
        preX = numpy.reshape(preX,(preX.size))
        preX=list(preX)
        prePredict=list(prePredict[-1])
        #print(type(preX))
        #print(prePredict[-1])
        #prePredict = numpy.reshape(prePredict[-1],(1))
        preY.append(prePredict[-1])
        preX.append(prePredict[-1])
        #print(preX)
        preX = numpy.array(preX[-look_back:])
        if shape==0:
            preX = numpy.reshape(preX, (1, 1, len(preX)))
        elif shape==1:
            preX = numpy.reshape(preX, (1, len(preX),1))

    preY = scaler.inverse_transform([preY])

    #print(preY)
    #########################################################################
    #print(testPredict)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])


    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.5f RMSE' % (trainScore))



    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    preY_Plot = numpy.empty_like(dataset)
    preY_Plot[:, :] = numpy.nan
    preY_Plot[-24:, :] = preY.T



    plot_num=24
    preScore = math.sqrt(mean_squared_error(scaler.inverse_transform(dataset)[-plot_num:,0], preY_Plot[-plot_num:,0]))
    print('Pre Score: %.5f RMSE' % (preScore))

    mapeScore_test,mape_list=MAPE(scaler.inverse_transform(dataset)[-plot_num:], preY_Plot[-plot_num:])
    #mapeScore_test,mape_list=MAPE(dataset[-96:], scaler.transform(preY_Plot[-96:]))
    print('MAPE_avg Test Score: %.5f MAPE' % (mapeScore_test))
    print(mape_list)


    #mapeScore_test2,mape_list2=MAPE2(dataset[-96:], scaler.transform(preY_Plot[-96:]))
    mapeScore_test2,mape_list2=MAPE2(scaler.inverse_transform(dataset)[-plot_num:], preY_Plot[-plot_num:])
    print('MAPE_real Test Score: %.5f MAPE' % (mapeScore_test2))
    print(mape_list2)


    if data_output == True:
        doc = os.listdir()
        if '数据输出' not in doc:
            os.makedirs('数据输出')
        else:
            shutil.rmtree('数据输出')
            os.makedirs('数据输出')

        trainPredictPlot_output= open('数据输出/trainPredictPlot_output.txt','a')
        for i in trainPredictPlot:
            if math.isnan(i[0]) == False:
                trainPredictPlot_output.write(str(i[0])+'\n')
        trainPredictPlot_output.close()

        train_real_output= open('数据输出/train_real_output.txt','a')
        for i in scaler.inverse_transform(dataset)[look_back:len(trainPredict)+look_back]:
            if math.isnan(i[0]) == False:
                train_real_output.write(str(i[0])+'\n')
        train_real_output.close()

        pre_output= open('数据输出/pre_output.txt','a')
        for i in preY_Plot[-plot_num:]:
            if math.isnan(i[0]) == False:
                pre_output.write(str(i[0])+'\n')
        pre_output.close()

        mape_avg_output= open('数据输出/mape_avg.txt','a')
        for i in mape_list:
            mape_avg_output.write(str(i)+'\n')
        mape_avg_output.close()

        mape_real_output= open('数据输出/mape_real.txt','a')
        for i in mape_list2:
            mape_real_output.write(str(i)+'\n')
        mape_real_output.close()

        mape_output = open('数据输出/误差.txt','w')
        mape_output.write('MAPE_avg:'+str(mapeScore_test)+'\n'+'MAPE:'+str(mapeScore_test2)+'\n'+"RMSE:"+str(preScore))
        #mape_output.write("RMSE:"+str(preScore))
        mape_output.close()



    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset),label='true')
    plt.plot(trainPredictPlot,label='trainPredict')
    plt.plot(preY_Plot,label='preY')
    plt.legend()
    if data_output == True:
        plt.savefig('数据输出/predict.png')
    if auto_shutdown!=True:
        plt.show()
    plt.close()


    # fig2 = plt.figure(facecolor='white')
    # ax2 = fig2.add_subplot(111)
    # plt.plot(mape_list, label='MAPE_avg')
    # plt.plot(mape_list2, label='MAPE')
    # plt.legend()
    # if data_output == True:
    #     plt.savefig('数据输出/mape.png')
    # #plt.show()
    # plt.close()

    plt.plot(history.history['loss'],label='loss')
    #plt.plot(history.history['val_loss'],label='val_loss')
    # plt.plot(history.history['acc'],label='acc')
    # plt.plot(history.history['val_acc'],label='val_acc')
    #plt.plot(history.history['mean_absolute_percentage_error'],label='mape')
    plt.legend()
    #plt.show()
    plt.close()

    if auto_shutdown ==True:
        thd=threading.Thread(target=exit)
        thd.daemon=True
        thd.start()

        for i in range(16):
            time.sleep(1)
            print('%d秒后自动休眠计算机，按y终止。'%(15-i))
            if i ==15:
                shutdown()