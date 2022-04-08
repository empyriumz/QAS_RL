# 2022 03 24
# PennyLane Quantum classifier
# Training not successful, WHY?
from custom_torch_dataset_class import MakeCircles
from custom_torch_dataset_class import MakeBlobs


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pennylane as qml


dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
dtypeInteger = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# define the QNode
n_qubits = 4
n_classes = 4
n_layers = 12

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def Qcircuit(params, angles):
    # 2022 03 26: Angle Embedding sometimes is not enough (sklearn make-circles dataset)
    # qml.AngleEmbedding(angles, wires=range(n_qubits))
    # qml.BasicEntanglerLayers(params, wires=range(n_qubits), rotation=qml.RZ)

    # 2022 03 26: Try our good old days angle RY RZ embedding..
    for i in range(n_qubits):
        qml.RY(angles[i, 0], wires=i)
        qml.RZ(angles[i, 1], wires=i)

    qml.StronglyEntanglingLayers(weights=params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_classes)]


class VQCTorch(nn.Module):
    def __init__(self, circuit, num_layers=4, num_qubits=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.circuit = circuit
        self.q_params = nn.Parameter(torch.randn(self.num_layers, 1))
        #self.q_params = nn.Parameter(0.01 * torch.randn(self.num_layers, self.num_qubits, 4))
     
    def get_angles_atan(self, in_x):
        return torch.stack(
            [torch.stack([torch.atan(item), torch.atan(item ** 2)]) for item in in_x]
        )

    def forward(self, batch_item):
        score_batch = []

        for single_item in batch_item:
            # res_temp = self.get_angles_atan(single_item)
            # print(single_item)
            # print(res_temp)
            # print(qml.draw(Qcircuit, expansion_strategy="device")(self.q_params, res_temp))
            res_temp = self.circuit(single_item, self.q_params)
            # print(res_temp)
            # q_out_elem = vqc.forward(res_temp)
            # print(q_out_elem)

            # clamp = 1e-9 * torch.ones(2).type(dtype).to(device)
            # clamp = 1e-9 * torch.ones(2)
            # normalized_output = torch.max(q_out_elem, clamp)
            # score_batch.append(normalized_output)
            score_batch.append(res_temp)

        scores = torch.stack(score_batch).view(len(batch_item), n_classes)

        return scores


def main():
    # Prepare the dataset
    # dataset = MakeCircles()
    dataset = MakeBlobs()
    data_loader_train = DataLoader(dataset, batch_size=50, shuffle=True)

    # Build the model
    # This version of model does not use batch input
    # batch is handled in the Optimization function
    circuit = Qcircuit()
    HybridVQC = VQCTorch(circuit=circuit, num_layers=n_layers, num_qubits=n_qubits)
    # print(HybridVQC.state_dict())

    # opt = torch.optim.RMSprop(HybridVQC.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    # print(HybridVQC.forward(X[0]))

    # Training
    # 2022 03 24: Need to use PyTorch DataLoader

    num_epochs = 100
    learn_rate = 1e-2
    batch_size = 32
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(HybridVQC.parameters(), lr=learn_rate)
    epoch_list = []

    for epoch in range(1, num_epochs + 1):
        epoch_list.append(epoch)
        running_loss = 0.0
        running_acc = 0.0
        counter = 0
        print("###epoch {} ###".format(epoch))

        for idx, (inputs, labels) in enumerate(data_loader_train):

            scores = HybridVQC.forward(inputs)
            _, preds = torch.max(scores, 1)
            loss = loss_fun(scores, labels)
            print(loss)

            # if check_fc_grad:
            # 	mix_model.fc.register_backward_hook(hook_fn_backward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item()
                running_acc += accuracy
                running_loss += loss
            if idx % (int(len(data_loader_train) / 5)) == 0:
                print("*" * int(counter) + "-" * int(20 - counter))
                print("running acc : {}".format(running_acc / ((idx + 1) * batch_size)))
                print("running loss : {}".format(running_loss / ((idx + 1))))
                counter += 1


if __name__ == "__main__":
    main()
