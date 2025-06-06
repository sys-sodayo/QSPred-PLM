from model import CNN
from getdata import  case_studies_x, case_studies_y
import torch
from torch.utils.data import TensorDataset, DataLoader


def to_test_model(x, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(64, 1280, 2).to(device)
    model.load_state_dict(torch.load("../model/IQSPred-PLM.pt"))
    model.eval()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    pr_list,test_predictions = [],[]

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)

            test_predictions.extend(preds.cpu().tolist())
            pr_list.extend(outputs[:, 1].cpu().numpy())
        return y, test_predictions


if __name__ == "__main__":
    y,test_predictions = to_test_model(case_studies_x, case_studies_y)
    for i in range(len(test_predictions)):
        print(y[i], test_predictions[i])



