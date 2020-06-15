from flask import Flask,render_template, send_from_directory, request
import os,time
from predictions import classification_nb,classification_svm,classification_cnn,classification_lstm
import pandas as pd
from text_sim import get_sim_tfidf
from search_for import compare_all_dataset
app = Flask(__name__)
sign_in_list = pd.read_csv('dataset/signin.csv')

@app.route('/',methods=["POST","GET"])
def main():
	return render_template('login.html')

@app.route('/back',methods=["POST","GET"])
def back():
    return render_template('middle_page.html')
    
@app.route('/forward',methods=["POST","GET"])
def forward():
    signin_button = request.form['signin_button']
    if signin_button == 'signin':

        signin_email = request.form['signin_email']
        signin_password = request.form['signin_password']
        email_list = sign_in_list['email'].tolist()
        password_list = sign_in_list['password'].tolist()
        if signin_email in email_list:
            if str(password_list[email_list.index(signin_email)]) == str(signin_password):
                return render_template('middle_page.html')
            else:
                msg = 'Wrong Password'
                return render_template('login.html',msg=msg)
        msg = 'Wrong username'        
        return render_template('login.html',msg=msg)
    else:
        return render_template('signup.html')

@app.route('/forward_signup',methods=["POST","GET"])
def forward_signup():
    global sign_in_list
    signup_name = request.form['signup_name']
    signup_email = request.form['signup_email']
    signup_password = request.form['signup_password']
    sign_in_list = sign_in_list.append({'email':signup_email,'password':signup_password,'name':signup_name},ignore_index=True)
    sign_in_list.to_csv('dataset/signin.csv',)
    return render_template('login.html')

@app.route('/cnn',methods=["POST","GET"])
def cnn():
    yhat,input_text = '',''
    if request.method == 'POST':
        input_text = request.form['classification_message']
        yhat = classification_cnn(input_text)
    return render_template('text_classification.html',ouputtext=input_text,
                            classification_type='cnn',classfication_response=yhat)

@app.route('/lstm',methods=["POST","GET"])
def lstm():
    yhat,input_text = '',''
    if request.method == 'POST':
        input_text = request.form['classification_message']
        yhat = classification_lstm(input_text)
    return render_template('text_classification.html',ouputtext=input_text,
                            classification_type='lstm',classfication_response=yhat)

@app.route('/svm',methods=["POST","GET"])
def svm():
    yhat,input_text = '',''
    if request.method == 'POST':
        input_text = request.form['classification_message']
        yhat = classification_svm(input_text)
    return render_template('text_classification.html',ouputtext=input_text,
                            classification_type='svm',classfication_response=yhat)

@app.route('/naive_bayes',methods=["POST","GET"])
def naive_bayes():
    yhat,input_text = '',''
    if request.method == 'POST':
        input_text = request.form['classification_message']
        yhat = classification_nb(input_text)
    return render_template('text_classification.html',ouputtext=input_text,
                            classification_type='naive_bayes',classfication_response=yhat)

@app.route('/text_similarity',methods=["POST","GET"])
def text_similarity():
    yhat,text1,text2 = '','',''
    if request.method == 'POST':
        text1 = request.form['sim_message1']
        text2 = request.form['sim_message2']
        yhat = get_sim_tfidf(text1,text2)
    return render_template('text_similarity.html',text1=text1,text2=text2,sim_percentage=yhat)

@app.route('/text_similarity_db',methods=["POST","GET"])
def text_similarity_db():
    yhat,text1,text2 = '','',''
    if request.method == 'POST':
        text1 = request.form['sim_message1']
        yhat = compare_all_dataset(text1)
    return render_template('text_similarity_db.html',text1=text1,sim_percentage=yhat)

@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)    

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory('fonts', path)

@app.route('/vendor/<path:path>')
def send_vendor(path):
    return send_from_directory('vendor', path)    

if __name__ == '__main__':
    app.run(debug = True)