from  flask import Flask, render_template,request
import pickle

filename = 'sentiment-analysis-model.pkl'
classifier = pickle.load(open(filename,'rb'))
cv = pickle.load(open('vectorize-text.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect = cv.transform(data).toarray()
        my_pred = classifier.predict(vect)
        return render_template('prediction.html',prediction=my_pred)

if __name__ == '__main__':
    app.run(debug=True)
