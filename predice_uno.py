import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%matplotlib inline

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

#Definimos una prob por cada componente de acuerdo a los mu y sigmas
def prob(x, vector, mu, sigma):
    Ps=1
    for i in x:
        comp = np.dot(i,vector)
        Ps=Ps*1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-mu)/sigma)**2) 
    return Ps

sigmas = np.zeros((10,5))
mus = np.zeros((10,5))
for i in range(10):
    x_t = x_train[y_train == i]
    for j in range(5):
        mus[i,j] = np.mean(x_t @ vectores[:,j])
        Pm = 0
        for sig in np.linspace(0.001,20,1000):
            Pp = prob(x_t, vectores[:,j], mus[i,j], sig)
            if Pp > Pm:
                Pm = Pp
                sigmas[i,j] = sig
                
def probdig(x, dig):
    Probl = np.zeros(5)
    for i in range(5):
        comp = np.dot(x, vectores[:,i])
        Probl[i] = 1/(sigmas[dig,i]*np.sqrt(2*np.pi))*np.exp(-1/2*((comp-mus[dig,i])/sigmas[dig,i])**2)
    P = np.prod(Probl)
    return P

def predict(x):
    Probs = np.zeros(10)
    for i in range(10):
        Probs[i] = probdig(x, i)
    return np.argmax(Probs)


plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)

N1 = 1257
Predict1 = np.zeros(N1)
Etiqueta1 = np.zeros(N1)
for muestra in range(N1):
    xm = x_train[muestra]
    Predict1[muestra] = predict(xm)==1
    Etiqueta1[muestra] = y_train[muestra] ==1
#print(sum(Etiqueta == Predict)/N)

#for i in range(N):
#    print(Etiqueta[i], Predict[i])
    
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Etiqueta1, Predict1)
P1 = cm1[1][1] / (cm1[1][1] + cm1[0][1])
R1 = cm1[1][1] / (cm1[1][1] + cm1[1][0])
f_11 = 2*P1*R1/(P1+R1)

plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title(r'Train $F_1=$%0.3f' %f_11)
plt.ylabel('True')
plt.xlabel('Predict')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j])+" = "+str(cm1[i][j]))
#plt.show()


N = 540
Predict = np.zeros(N)
Etiqueta = np.zeros(N)
for muestra in range(N):
    xm = x_test[muestra]
    Predict[muestra] = predict(xm)==1
    Etiqueta[muestra] = y_test[muestra] ==1
#print(sum(Etiqueta == Predict)/N)

cm = confusion_matrix(Etiqueta, Predict)
P = cm[1][1] / (cm[1][1] + cm[0][1])
R = cm[1][1] / (cm[1][1] + cm[1][0])
f_1 = 2*P*R/(P+R)
plt.subplot(1, 2, 2)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title(r'Test $F_1=$%0.3f' %f_1)
plt.ylabel('True')
plt.xlabel('Predict')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))

plt.savefig('matriz_de confusión.png')
plt.show()