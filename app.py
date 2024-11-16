# app.py
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run-detect', methods=['POST'])
def run_detect():
    # Extract parameters from the request
    data = request.get_json()
    source = data.get('source', 'path/to/your/images')
    weights = data.get('weights', 'runs/train/exp/weights/best.pt')
    img_size = data.get('img_size', 640)
    conf_thres = data.get('conf_thres', 0.25)

    # Construct the command to run detect.py
    command = [
        'python', 'detect.py',
        '--weights', weights,
        '--source', source,
        '--img', str(img_size),
        '--conf', str(conf_thres)
    ]

    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Return the output
        return jsonify({'output': result.stdout}), 200
    except subprocess.CalledProcessError as e:
        # Handle errors
        return jsonify({'error': e.stderr}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
