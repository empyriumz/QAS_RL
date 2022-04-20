import torch
import sys
import numpy as np
import curricula
import pennylane as qml

from custom_torch_dataset_class import MakeBlobs, Iris, Digits
from pennylane_classifier_base import VQCTorch

"""
TODO:
1. We would like to demonstrate that our agent will be generalizable,
i.e., work for different kinds of dataset & different # of qubits
2. Moving threshold?
"""


class CircuitEnv:
    def __init__(self, conf, device, RANDOM_SEED=0):
        self.config = conf
        self.num_qubits = self.config["env"]["num_qubits"]
        self.num_layers = self.config["env"]["num_layers"]
        self.num_classes = self.config["env"]["num_classes"]
        self.n_samples_train = self.config["classifier"]["num_samples_train"]
        self.n_samples_val = self.config["classifier"]["num_samples_val"]
        self.min_accuracy = float(self.config["classifier"]["min_accuracy"])
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        # self.circuit = qml.device("default.qubit", wires=self.num_qubits)
        self.fake_min_accuracy = (
            self.config["env"]["fake_min_accuracy"]
            if "fake_min_accuracy" in self.config["env"].keys()
            else None
        )
        self.fn_type = self.config["env"]["fn_type"]

        # If you want to run agent from scratch without *any* curriculum just use the setting with
        # normal curriculum and set self.config[episodes] = [1000000]
        # self.curriculum_dict = {}

        # self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[
        #     self.config["env"]["curriculum_type"]
        # ](self.config["env"], target_energy=min_eig)

        self.device = device
        # self.done_threshold = self.config["env"]["accept_err"]

        sys.stdout.flush()
        self.state_size = 4 * self.num_layers  # 2 for CNOT, 2 for rotation
        self.actual_layer = 0
        self.prev_accuracy = None
        self.accuracy = 0.0
        ## n * (n -1) for possible CNOT configurations, n * 3 for possible rotations
        self.action_size = self.num_qubits * (self.num_qubits + 2)
        # n_samples_val = 144
        # self.RANDOM_SEED = RANDOM_SEED
        # id_list = list(range(720))
        # rng = np.random.default_rng(self.RANDOM_SEED)
        # val_id = list(rng.choice(id_list, size=n_samples_val, replace=False))
        # train_id = list(set(id_list).difference(val_id))
        # self.dataset_train = Digits(train_id, n_class=self.num_classes)
        # self.dataset_val = Digits(val_id, n_class=self.num_classes)

        n_samples_val = 32
        self.RANDOM_SEED = RANDOM_SEED
        id_list = list(range(150))
        rng = np.random.default_rng(self.RANDOM_SEED)
        val_id = list(rng.choice(id_list, size=n_samples_val, replace=False))
        train_id = list(set(id_list).difference(val_id))
        self.dataset_train = Iris(train_id)
        self.dataset_val = Iris(val_id)

        # self.dataset_train = MakeBlobs(n_samples=self.n_samples_train)
        # self.dataset_val = MakeBlobs(n_samples=self.n_samples_val)
        self.batch_size = self.config["classifier"]["batch_size"]
        self.data_loader_train = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_loader_val = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False
        )
        sample_x, _ = next(iter(self.data_loader_train))
        self.test_sample = sample_x[0]

    def step(self, action):

        """
        Action is performed on the first empty layer.
        Variable 'actual_layer' points last non-empty layer.
        """
        next_state = self.state.clone()

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """

        next_state[0][self.actual_layer] = action[0]
        next_state[1][self.actual_layer] = (action[0] + action[1]) % self.num_qubits

        ## state[2] corresponds to number of qubit for rotation gate
        next_state[2][self.actual_layer] = action[2]
        next_state[3][self.actual_layer] = action[3]

        self.state = next_state.clone()
        self.accuracy = self.get_accuracy()
        self.state = next_state.clone()

        # if self.accuracy < self.curriculum.lowest_accuracy and train_flag:
        #     self.curriculum.lowest_.accuracy = copy.copy(self.accuracy)
        rwd = self.reward_func(self.accuracy)
        self.prev_accuracy = np.copy(self.accuracy)

        self.actual_layer += 1
        accuracy_done = int(self.accuracy > self.min_accuracy)
        layers_done = self.actual_layer == self.num_layers
        done = int(accuracy_done or layers_done)
        # if done:
        #     self.curriculum.update_threshold(accuracy_done=accuracy_done)
        #     self.done_threshold = self.curriculum.get_current_threshold()
        #     self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(
        #         self.curriculum
        #     )
        print(
            "# of layers {}, Accuracy: {:.2f}, Reward: {}".format(
                self.actual_layer, self.accuracy, rwd
            )
        )
        # print(qml.draw(self.circuit)(self.test_sample, self.model.q_params))
        return (
            next_state.view(-1).to(self.device),
            torch.tensor(rwd, dtype=torch.float32, device=self.device),
            done,
        )

    def reset(self):
        """
        Returns empty state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with CONTROL gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with NOT gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        ## state_per_layer: (Control_qubit, NOT_qubit, R_qubit, R_axis, R_angle)
        ## the initialization means no gates applied [num_qubits, 0, num_qubits, 0, 0]
        controls = self.num_qubits * torch.ones(self.num_layers)
        nots = torch.zeros(self.num_layers)
        rotations = self.num_qubits * torch.ones(self.num_layers)
        generators = torch.zeros(self.num_layers)
        # angles = torch.zeros(self.num_layers)

        state = torch.stack((controls, nots, rotations, generators))
        self.state = state

        # self.make_circuit(state)
        self.actual_layer = 0

        # self.curriculum = copy.deepcopy(
        #     self.curriculum_dict[str(self.current_bond_distance)]
        # )
        # self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.prev_accuracy = (
            1 / self.num_classes
        )  # using the baseline of random classifier

        return state.view(-1).to(self.device)

    def make_circuit(self, features, weights):
        qml.AmplitudeEmbedding(
            features=features, wires=range(self.num_qubits), pad_with=0, normalize=True
        )
        self.layer(weights)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_classes)]

    def layer(self, thetas):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control qubit
        if thetas[1, i] == 0 then there is no rotation gate on the NOT qubit
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.detach().cpu().numpy()

        for i in range(self.num_layers):
            if state[0][i] != self.num_qubits:
                qml.CNOT(wires=[int(state[0][i]), int(state[1][i])])
            elif state[2][i] != self.num_qubits:
                if int(state[3][i]) == 1:
                    qml.RX(thetas[i][0], wires=int(state[2][i]))
                elif int(state[3][i]) == 2:
                    qml.RY(thetas[i][0], wires=int(state[2][i]))
                elif int(state[3][i]) == 3:
                    qml.RZ(thetas[i][0], wires=int(state[2][i]))

    def get_accuracy(self):
        self.circuit = qml.QNode(self.make_circuit, self.dev, interface="torch")
        self.model = VQCTorch(
            self.circuit,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            num_qubits=self.num_qubits,
        )
        accuracy = self.train()
        return accuracy

    def train(self):
        num_epochs = self.config["classifier"]["num_epochs"]
        learning_rate = self.config["classifier"]["learning_rate"]

        loss_fun = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_accuracy = 0.0
        prev_accuracy = 0.0
        early_stopping = 0
        for _ in range(num_epochs):
            running_acc = 0.0
            running_loss = 0.0
            for inputs, labels in self.data_loader_train:
                # inputs = inputs.to(self.device)
                scores = self.model(inputs)
                _, preds = torch.max(scores, 1)
                loss = loss_fun(scores, labels)
                optimizer.zero_grad()
                ## in case if the circuit only has CNOT gates, i.e., no trainable parameters
                try:
                    loss.backward()
                    optimizer.step()
                except:
                    break
                accuracy = torch.sum(preds == labels).detach().cpu().numpy()
                running_loss += loss.detach().cpu().numpy()
            # print("running loss {:.4f}".format(running_loss / len(self.dataset_train)))
            with torch.no_grad():
                for inputs, labels in self.data_loader_val:
                    scores = self.model(inputs)
                    _, preds = torch.max(scores, 1)
                    accuracy = torch.sum(preds == labels).detach().cpu().numpy()
                    running_acc += accuracy

            accuracy = running_acc / len(self.dataset_val)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
            if best_accuracy > self.min_accuracy:
                return best_accuracy
            if accuracy <= prev_accuracy:
                early_stopping += 1
            prev_accuracy = accuracy
            if early_stopping > 4:
                # print("Early stopping activated")
                return best_accuracy

        return best_accuracy

    def reward_func(self, accuracy):
        max_depth = self.actual_layer == self.num_layers
        if accuracy >= self.min_accuracy:
            reward = 5.0 - 0.01 * self.actual_layer
        elif max_depth:
            reward = -5.0
        else:
            reward = np.clip(
                (accuracy - self.prev_accuracy) / abs(self.prev_accuracy + 1e-6)
                - 0.01 * self.actual_layer,
                -1.5,
                1.5,
            )
        return reward
