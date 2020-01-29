from random import randint as ri
from multiprocessing import Process,Pipe
import tensorflow as tf
import numpy as np
import functools as tool
import time
import copy
import pickle 
import matplotlib.pyplot as plt

class main:
    def __init__(self):
        (self.x_train,self.y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #prepear data to be MxN [ [[],[],[]] , [[],[],[]] ]
        
        self.cl_n = 10
        self.pix_l = len(self.x_train[0])*len(self.x_train[0][0])
        self.exmp = len(self.y_train)
        
        self.delt = 80
        self.lam = 0.4
        #self.eps = 0.000001
        self.nu  = 0.000001
        self.aprox = 0.0001
        
        start_time = time.time()
        self.fini = False
        self.proxi = 0
        self.initial()
        self.train()
        print("--- %s seconds ---" % (time.time() - start_time))
        
    def initial(self):
        self.w = list([[round(ri(1,10)*(10**(-5)),6) for y in range(self.pix_l)] for u in range(self.cl_n)])
        self.b = list([round(ri(1,10)*(10**(-4)),5) for y in range(self.cl_n)])
        self.x_train = np.array(self.x_train).tolist()
        self.y_train = np.array(self.y_train).tolist()

    
    def train(self):
        self.depth = 0
        while True :
            with open("net",'wb') as fi:
                pickle.dump(net(self.w,self.b),fi)
            if self.fini:
                break
            else:
                print("_________________")
                print (self.depth)
                self.iteration(self.depth)
            
            
    def iteration(self,k):
        self.data = [[tool.reduce(lambda x,y:x+y ,np.array(self.x_train[u]).tolist()),self.y_train[u]] for u in range((k*self.delt)%self.exmp,(k*self.delt)%self.exmp+self.delt)]
        new = self.fast_gradient()
        a = Pipe()
        self.loss(a[1],self.w,self.b)
        b = Pipe()
        self.loss(b[1],new[0],new[1])
        e1 = a[0].recv()
        e2 = b[0].recv()
        a[0].close()
        b[0].close()
        q = self.exmp/self.delt
        if (k%(0.1*q) == 0):
            print (k," e1",e1,"  e2",e2,"  e1 > e2",e1>e2," score ",self.proxi)
        self.w = new[0]
        self.b = new[1]
        if k%q > 0:
            if ((e1-self.aprox) > (e2+self.aprox)):
                self.proxi = self.proxi + 1
            else:
                self.proxi = 0
            self.iteration(k+1)
            
        else:
            if ((e1-self.aprox) > (e2+self.aprox)):
                self.proxi = self.proxi + 1
            else:
                self.proxi = 0
            if self.proxi > (2*q):
                self.fini = True
                print ("~~~------~~~")
            self.depth = k+1

        
    
    def fast_gradient(self):
        a = Pipe()
        self.loss(a[1],self.w,self.b,False)
        n_v,n_b = a[0].recv()
        a[0].close()
        new_w = np.array(self.w) - (self.nu*np.array(n_v))
        new_b = np.array(self.b) - (self.nu*np.array(n_b))
        return (new_w,new_b)
        
    def gradient(self):
        n_v = [[] for u in range(self.cl_n)]
        n_b = []
        def pm(x,g,u,l):
            ch_w = copy.deepcopy(l)
            if g:
                ch_w[x][u] = ch_w[x][u] + self.eps
            else:
                ch_w[x][u] =ch_w[x][u] - self.eps
            return ch_w
        
        for u in range(self.pix_l):
            if (u%100 == 0):
                print(u)
            for g in [True,False]:
                conn = [Pipe() for v in range(self.cl_n)]
                g_w = (pm(x,g,u,self.w) for x in range(self.cl_n))
                proc = [Process(target=self.loss, args=(conn[u][1],next(g_w),self.b)) for u in range(self.cl_n)]
                (lambda : [u.start() for u in proc])()
                (lambda : [u.join() for u in proc])()
                res = [t[0].recv() for t in conn]
                for f in range(len(res)):
                    if g:
                        n_v[f].append(res[f])
                    else:
                        n_v[f][u]= (n_v[f][u] - res[f])/(2*self.eps)
                (lambda : [u[0].close() for u in conn])()
                                
        for j in [True,False]:
            conn = [Pipe() for v in range(self.cl_n)]
            g_b = (pm(0,j,x,[self.b])[0] for x in range(self.cl_n))
            proc = [Process(target=self.loss, args=(conn[u][1],self.w,next(g_b))) for u in range(self.cl_n)]
            (lambda : [u.start() for u in proc])()
            (lambda : [u.join() for u in proc])()
            res = [t[0].recv() for t in conn]
            for f in range(len(res)):
                if j:
                    n_b.append(res[f])
                else:
                    n_b[f]= (n_b[f] - res[f])/(2*self.eps)
            (lambda : [u[0].close() for u in conn])()
        
        self.fast_gradient()
        new_w = np.array(self.w) - (self.nu*np.array(n_v))
        new_b = np.array(self.b) - (self.nu*np.array(n_b))
        return (new_w,new_b)
            
    def loss(self,c,wu,bu,loss = True):
        con = [Pipe() for v in range(self.delt)]
        proc = [Process(target=self.count, args=(con[u][1],self.data[u][0],self.data[u][1],wu,bu,loss)) for u in range(self.delt)]
        (lambda : [q.start() for q in proc])()
        (lambda : [e.join() for e in proc])()
        res = [t[0].recv() for t in con]
        (lambda : [u[0].close() for u in con])()
        if loss:
            res = -1* sum(res)
            c.send(self.regulation(res,wu,bu))
        else:
            r_w = [t[0] for t in res]
            r_b = [t[1] for t in res]
            g_w = np.array(tool.reduce(lambda x,y:x+y,np.array(r_w))+(2*self.lam)*np.array(wu)).tolist()
            g_b = np.array(tool.reduce(lambda x,y:x+y,np.array(r_b))+(2*self.lam)*np.array(bu)).tolist()
            c.send((g_w,g_b))
        c.close()

    def regulation(self,num,we,be):
        sw = tool.reduce(lambda x,y:x+y**2, [0]+(tool.reduce(lambda x,y:x+y,we)))
        sb = tool.reduce(lambda x,y:x+y**2, [0]+be)
        return num + self.lam *(sw+sb)
    
    def count(self,conn,lis,ind,wi,bi,loss):
        li = np.add(np.dot(lis,np.transpose(wi)),bi).tolist()
        dwn = tool.reduce(lambda x,y: x + np.e**(y), [0]+li)
        if loss:
            up = np.e**(li[ind])
            conn.send(np.log(up/dwn))
        else:
            softmax = [(np.e**(li[i]))/dwn for i in range(len(li))]
            softmax[ind] = softmax[ind] -1
            ans = np.dot(np.transpose([softmax]),[lis]).tolist()
            conn.send((ans,softmax))
        conn.close()
        
class net:
    def __init__(self,w,b):
        (x_train,y_train),(self.x_test,self.y_test) = tf.keras.datasets.mnist.load_data()
        self.w = w
        self.b = b
        self.x_test = np.array(self.x_test).tolist()
        self.y_test = np.array(self.y_test).tolist()
        
    def prdct(self,v):
        v = tool.reduce(lambda x,y:x+y ,np.array(v).tolist())
        li = np.add(np.dot([v],np.transpose(self.w)),self.b)
        li = np.array(li).tolist()[0]
        su = tool.reduce(lambda x,y: x + np.e**y, [0]+li)
        li = list(map(lambda x:np.e**x/su,li))
        return li.index(max(li))
    
    def check(self,k):
        print(self.prdct(self.x_test[k]))
        self.show(self.x_test[k])
    def check_all(self):
        a = len(self.y_test)
        k = 0
        for y in range(len(self.y_test)):
            p = self.prdct(self.x_test[y])
            if p == self.y_test[y]:
                k = k + 1
        
        print (k, " of ",a," % ",(k*100)/a) 
        '''
        8864  of  10000  %  88.64
        
        15750  e1 24.49899470351744   e2 2.9398495923424908   e1 > e2 True  score  6
        '''
            
    def show(self,num):
        plt.imshow(num, cmap='Greys')
        plt.show()
        
        
        
    
if __name__ == "__main__":
    #cl = main()
    with open("net",'rb') as f:
        n = pickle.load(f)
    n.check_all()

    
    
    
        
