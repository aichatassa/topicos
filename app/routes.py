from flask import render_template, request, send_file
from app import app

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64

import tempfile
loop = asyncio.get_event_loop()
executor = ThreadPoolExecutor(1)
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html', knn_params=['n_neighbors'], svm_params=['C'], mlp_params=['hidden_layer_sizes', 'max_iter'], dt_params=['max_depth', 'min_samples_split'])

@app.route('/train_test', methods=['POST'])
def train_test():
    classifier_name = request.form.get('classifier')
    param1 = float(request.form.get('param1'))
    param2 = float(request.form.get('param2'))
    param3 = float(request.form.get('param3'))

    X, y = np.random.rand(100, 3), np.random.randint(0, 2, 100)

    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=int(param1))
        classifier_params = f'n_neighbors={int(param1)}'
    elif classifier_name == 'svm':
        classifier = SVC(kernel='linear', C=param1)
        classifier_params = f'kernel=linear, C={param1}'
    elif classifier_name == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=(int(param1),), max_iter=int(param2))
        classifier_params = f'hidden_layer_sizes=({int(param1)}), max_iter={int(param2)}'
    elif classifier_name == 'dt':
        classifier = DecisionTreeClassifier(max_depth=int(param1), min_samples_split=int(param2))
        classifier_params = f'max_depth={int(param1)}, min_samples_split={int(param2)}'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    confusion_matrix_image = run_blocking_io(plot_confusion_matrix, y_test, y_pred)

    return render_template('result.html', accuracy=accuracy, f1_score=f1_score,
                           confusion_matrix_image=confusion_matrix_image,
                           classifier_params=classifier_params)

def run_blocking_io(func, *args):
    result = loop.run_in_executor(executor, func, *args)
    return loop.run_until_complete(result)

def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    base64_encoded_image = base64.b64encode(img_buf.getvalue()).decode('utf-8')

    return base64_encoded_image

