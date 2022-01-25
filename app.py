import os
import json
import webbrowser
import subprocess
from threading import Lock

from flask import (
    flash,
    Flask,
    render_template,
)
from flask_socketio import SocketIO, emit

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crlt'

thread = None
thread_lock = Lock()

async_mode = None
socketio = SocketIO(app, async_mode=async_mode)

logging_dir = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on("tensorboard")
def tensorboard():
    if os.path.exists(logging_dir):
        commands = f"tensorboard --logdir {logging_dir} --port 6006"
        subprocess.call(commands, shell=True)
        message = "success"
    else:
        message = "failure"
    socketio.emit('tensorboard_response', {'message': message})


def training(training_params):
    # training_params = kwargs['training_params']
    training_params = json.loads(training_params)
    output_dir = training_params['output_dir']
    global logging_dir
    logging_dir = training_params['logging_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "traning_params.json"), "w") as fout:
        if isinstance(training_params, dict):
            json.dump(training_params, fout)
        else:
            json.dump(eval(training_params), fout)
    traning_params_dir = os.path.join(output_dir, "traning_params.json")
    commands = f"CUDA_VISIBLE_DEVICES=0 python main.py {traning_params_dir}"
    proc = subprocess.Popen(
        "/bin/bash",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )

    proc.stdin.write(commands.encode())
    proc.stdin.flush()
    proc.stdin.close()

    while True:
        # Real time stdout of subprocess
        line = proc.stdout.readline().decode()

        if line == "" and proc.poll() is not None:
            break

        socketio.emit('logging', {'data': line})


@socketio.event
def connection(message):
    print("Connection")


@socketio.on("run")
def run(message):
    training_params = message["data"]
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(
                training, training_params)
            flash("Start training!")
        else:
            flash('Please do not click repeatedly during training!')


if __name__ == '__main__':
    socketio.run(app)
