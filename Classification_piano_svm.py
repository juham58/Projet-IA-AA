from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file = np.load('pretrained_data.npy', allow_pickle=True).reshape(1)[0]


list_scores = []
list_confusion_matrix = []

for i in range(1000):


    data, target = shuffle(file['data'], file['target'])

    data_train = data[:300]
    target_train = target[:300]

    data_test = data[300:]
    target_test = target[300:]

    svm = SVC()

    svm.fit(data_train, target_train)

    score = svm.score(data_test, target_test)
    list_scores.append(score)



    prediction_test = svm.predict(data_test)


    vp = 0
    vn = 0
    fp = 0
    fn = 0

    for i, prediction in enumerate(prediction_test):
        if prediction == 1 and target_test[i] == 1:
            vp += 1
        elif prediction == 1 and target_test[i] == 0:
            vn += 1
        elif prediction == 0 and target_test[i] == 0:
            fp += 1
        elif prediction == 0 and target_test[i] == 1:
            fn += 1
       
    confusion_matrix = np.array([[vp, fp],[fn, vn]])
    
    list_confusion_matrix.append(confusion_matrix)


mean_score = np.mean(list_scores)
std_score = np.std(list_scores)
print(mean_score*100, std_score*100)

mean_confusion_matrix = np.mean(list_confusion_matrix, axis=0)*100/30

std_confusion_matrix = np.std(list_confusion_matrix, axis=0)*100/30

print(mean_confusion_matrix)
print(std_confusion_matrix)


ax = sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues')

ax.set_title('Matrice de confusion [%]');
ax.set_xlabel('Valeur prédite')
ax.set_ylabel('Vraie valeur ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Faux','Vrai'])
ax.yaxis.set_ticklabels(['Faux','Vrai'])

plt.show()

ax2 = sns.heatmap(std_confusion_matrix, annot=True, cmap='Blues')

ax2.set_title('Incertitude sur la matrice de confusion [%]');
ax2.set_xlabel('Valeur prédite')
ax2.set_ylabel('Vraie valeur ');

## Ticket labels - List must be in alphabetical order
ax2.xaxis.set_ticklabels(['Faux','Vrai'])
ax2.yaxis.set_ticklabels(['Faux','Vrai'])



## Display the visualization of the Confusion Matrix.
plt.show()



















