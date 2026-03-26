"""
FocusEye — ML Camera Filter Web App
Python Flask backend that serves the frontend.
All real-time ML (face mesh, hand tracking, body pose) runs
client-side via MediaPipe JS for zero-latency processing.
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("=" * 50)
    print("  FocusEye — ML Camera Filter App")
    print("=" * 50)
    print("  Open http://localhost:5000 in Chrome or Edge")
    print("  (Camera requires a modern browser)")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
