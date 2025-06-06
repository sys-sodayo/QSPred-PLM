import torch

test_pos = torch.load('../dataset/test set/test_pos_esm.pt')
test_neg = torch.load('../dataset/test set/test_neg_esm.pt')

test_x = torch.cat((test_neg, test_pos),0)
test_y = torch.tensor([0]*20 + [1]*20)

case_studies_x  = torch.load('../dataset/test set/case_studies_esm.pt')
case_studies_y = torch.tensor([1]*12)