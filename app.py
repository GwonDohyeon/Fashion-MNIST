from flask import Flask, request, jsonify, render_template
from modules.model import predict # 모델 로드 함수 임포트
import torch 
from PIL import Image
from flask_cors import CORS
app = Flask(__name__)
CORS(app) # CORS 허용
 # 모델 불러오기
@app.route('/')
def index():
 return render_template('index.html') # templates 폴더에 있는 index.html 반환
@app.route('/predict', methods=['POST'])
def predict1():
   if 'file' not in request.files:
       return jsonify({'error': 'No file uploaded'}), 400
   file=request.files['file']
   if file.filename=='':
      return jsonify({'error': 'No selected file'})
   if file:
 # 이미지를 전처리하고 모델로 예측
      result=predict(file)
    # 결과 반환
      return jsonify({'prediction': result})
   else:
      return jsonify({'error': 'Invalid file'})
   
if __name__=='__main__':
 app.run(debug=True)