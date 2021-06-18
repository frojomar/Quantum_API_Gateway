from flask import Flask, request, jsonify
from flask_cors import CORS
from estimation_model import write_feedback
from balancer import *
import os
from pathlib import Path


app = Flask(__name__)
CORS(app)

# #PREVIOUSLY EXECUTE: Configure credentials and region in the files ~/.aws/credentials and ~/.aws/config, respectively



@app.route('/execute', methods=["get"])
def get_execution_environment():
    content = request.json

    if content is None:
        return "", 400

    type_machine = content["type"]
    qubits = content["qubits"]
    priority = content["priority"]
    cost_threshold = float(content["cost_threshold"])
    shots = content["shots"]
    print(type(cost_threshold))
    if (type_machine or qubits or priority or cost_threshold or shots) is None:
        return "", 400

    try:

        best_machine, estimated_cost, estimated_time = get_recommended_machine(type_machine, qubits, priority, cost_threshold, shots)

        if estimated_time == INF:
            estimated_time = -1
        response = {
            "best_machine": best_machine,
            "estimated_time": float(estimated_time),
            "estimated_cost": estimated_cost
        }

        return jsonify(response), 201

    except ParamError as pe:
        return jsonify({"error": str(pe)}), 400

    except NotFoundMachine as nfm:
        return jsonify({"best_machine": None,
                        "reason": str(nfm)}), 404

    except Exception as e:
        print(e)
        return "", 500





@app.route('/feedback', methods=["post"])
def post_feedback():

    content = request.json

    if content is None:
        return "", 400

    machine = content["machine"]
    qubits = content["qubits"]
    #complexity = content["complexity"]
    shots = content["shots"]
    time = content["time"]
    day = content["day"]
    execution_time = content["execution_time"]

    if (machine or qubits  or shots or time or day or execution_time) is None:
        return "", 400

    try:
        write_feedback(machine, qubits, shots, time, day, execution_time)
        return "", 201
    except Exception as e:
        print(e)
        return "", 500



if __name__ == '__main__':
    print("[INIT] Executing load balancer... \n")
    print("[CONFIGURATION] Previously execute, install boto3 and configure credentials and region in the files ~/.aws/credentials and ~/.aws/config, respectively \n")

    home = str(Path.home())
    if(os.path.isfile(os.path.join(home,".aws/credentials"))):
        if (os.path.isfile(os.path.join(home,".aws/config"))):
            app.run(host="localhost", port=33888)
        else:
            print("\n [ERROR] Install boto3 (if not already done) and configure AWS region on ~/.aws/config \n")

    else:
        print("\n [ERROR] Install boto3 (if not already done) and configure AWS credentials on ~/.aws/credentials \n")
