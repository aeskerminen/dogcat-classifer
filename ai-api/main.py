from flask import Flask, request, jsonify
from flask_cors import CORS
from tester import predict
import os


app = Flask(__name__)
CORS(app=app)

# Define the upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Ok, we received and saved the file. Now run it through the AI and return the result.
        prediction = predict("model.pth", filepath)

        return jsonify({"message": f"File {file.filename} saved successfully", "file_path": filepath, "prediction": prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)
