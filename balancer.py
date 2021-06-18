from datetime import datetime

from braket.aws import AwsDevice

from estimation_model import predict_time

# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass

class ParamError(Error):
    """Raised when the value of param is not correct"""
    pass

class NotFoundMachine(Error):
    """Raised when the input value is too small"""
    pass

INF = 9999999999999999
MACHINES = [["riggeti_aspen8", "riggeti_aspen9", "ionq"],["dwave_advantage","dwave_dw2000"]] #[gate machines, annealing machines+]
TYPES = ["gate", "annealing"]

MACHINES_ARN= { "riggeti_aspen8":"arn:aws:braket:::device/qpu/rigetti/Aspen-8",
                "riggeti_aspen9":"arn:aws:braket:::device/qpu/rigetti/Aspen-9",
                "ionq":"arn:aws:braket:::device/qpu/ionq/ionQdevice",
                "dwave_advantage":"arn:aws:braket:::device/qpu/d-wave/Advantage_system1",
                "dwave_dw2000":"arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6"}

adiabatic_machines_arn= { }

LIMIT_QUBITS ={
    "riggeti_aspen8": INF,
    "riggeti_aspen9": INF,
    "ionq": 11,
}
COST_EXECUTION = {
    "riggeti_aspen8": 0.30,
    "riggeti_aspen9": 0.30,
    "ionq": 0.30,
    "dwave_advantage": 0.30,
    "dwave_dw2000": 0.30
}

COST_PER_SHOT = {
    "riggeti_aspen8": 0.00035,
    "riggeti_aspen9": 0.00035,
    "ionq": 0.01000,
    "dwave_advantage": 0.00019,
    "dwave_dw2000": 0.00019
}

def compute_cost(machine, shots):
    return COST_EXECUTION[machine] + (shots*COST_PER_SHOT[machine])


def get_time():
    now = datetime.now()
    return now.hour*3600+now.minute*60+now.second

def get_week_day():
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return days[datetime.today().weekday()]


def get_status(machine):
    arn = MACHINES_ARN[machine]
    device = AwsDevice(arn)
    return device.status


def get_response(machine, qubits, shots):
    cost = compute_cost(machine, shots)
    if machine in MACHINES[0]:
        type = TYPES[0]
    else:
        type = TYPES[1]
    waiting_time = predict_time(type, machine, qubits, shots, get_time(), get_week_day())
    return machine, cost, waiting_time

def get_cheaper(machines, qubits, shots):
    machine_sel = ""
    cost_sel = INF
    time_sel = INF
    for machine in machines:
        m, cost, time = get_response(machine, qubits, shots)
        if cost < cost_sel:
            machine_sel = m
            cost_sel = cost
            time_sel = time
    return machine_sel, cost_sel, time_sel

def get_faster(machines, qubits, shots):
    machine_sel = ""
    cost_sel = INF
    time_sel = INF
    for machine in machines:
        m, cost, time = get_response(machine, qubits, shots)
        if time <= time_sel:
            machine_sel = m
            cost_sel = cost
            time_sel = time
    return machine_sel, cost_sel, time_sel

def get_recommended_machine(type, qubits, priority, cost_threshold, shots):

    # 1. Choose machines per type
    if type not in TYPES:
        raise ParamError("Type of machine not allowed. Value must be one of these: ['gate', 'annealing']")
    else:
        if type == TYPES[0]:
            return choose_gate_machine(MACHINES[0], qubits, priority, cost_threshold, shots)
        else:
            return choose_annealing_machine(MACHINES[1], priority, cost_threshold, shots)


def choose_annealing_machine(machines, priority, cost_threshold, shots):
    # 2. Choose machine in function of cost's threshold

    if type(shots) != int or shots < 0:
        raise ParamError("Shots must be a integer greater to 0")
    if type(cost_threshold) not in [int, float] or cost_threshold < 0:
        raise ParamError("Cost threshold must be a integer or float")

    valid_machines = []
    for machine in machines:
        cost = compute_cost(machine, shots)
        if cost < cost_threshold:
            valid_machines.append(machine)

    if len(valid_machines) == 0:
        raise NotFoundMachine(
            "The cost's threshold is low. You need a higher threshold to be able to run your code on a compliant machine.")
    else:
        machines = valid_machines

    # 4. Filter only availables machines
    valid_machines = []
    for machine in machines:
        status = get_status(machine)
        if status=="ONLINE":
            valid_machines.append(machine)

    if len(valid_machines)==0:
        raise NotFoundMachine("There are not available machines that meet the requirements specified to that execution.")
    else:
        machines = valid_machines

    if len(machines) == 1:
        return get_response(machines[0], None, shots)
    # 3. Recommend a solution in basis of priority requirements
    else:
        if type(priority) is not int or priority < 0:
            raise ParamError("Priority must be a integer between 0 and 10")

        if priority <= 5:
            return get_cheaper(machines, None, shots)
        else:
            return get_faster(machines, None, shots)



def choose_gate_machine(machines, qubits, priority, cost_threshold, shots):

    # 2. Choose machine in function of qubits
    if type(qubits) != int or qubits < 0:
        raise ParamError("Qubits must be a integer greater to 0 (if gate-based execution) or equal to 0 or null (if adiabatic/annealing execution)")
    else:

        valid_machines = []
        for machine in machines:
            if LIMIT_QUBITS[machine] >= qubits:
                valid_machines.append(machine)

        if len(valid_machines)==0:
            raise NotFoundMachine("There are not available machines with so much qubits")
        else:
            machines = valid_machines
        # 3. Choose machine in function of cost's threshold

        if type(shots) is not int or shots < 0:
            raise ParamError("Shots must be a integer greater to 0")
        if type(cost_threshold) not in [int, float] or cost_threshold < 0:
            raise ParamError("Cost threshold must be a integer or float")

        valid_machines = []
        for machine in machines:
            cost = compute_cost(machine, shots)
            if cost < cost_threshold:
                valid_machines.append(machine)

        if len(valid_machines)==0:
            raise NotFoundMachine("The cost's threshold is low. You need a higher threshold to be able to run your code on a compliant machine.")
        else:
            machines = valid_machines

        # 4. Filter only availables machines
        valid_machines = []
        for machine in machines:
            status = get_status(machine)
            print(machine, status)
            if status=="ONLINE":
                print(machine, "selected")
                valid_machines.append(machine)

        if len(valid_machines)==0:
            raise NotFoundMachine("There are not available machines that meet the requirements specified to that execution.")
        else:
            machines = valid_machines

        if len(machines)==1:
            return get_response(machines[0], qubits, shots)

        # 5. Recommend a solution in basis of priority requirements
        else:
            if type(priority) is not int or priority < 0:
                raise ParamError("Priority must be a integer between 0 and 10")

            if priority<=5:
                return get_cheaper(machines, qubits, shots)
            else:
                return get_faster(machines, qubits, shots)