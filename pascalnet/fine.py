model_ft = models.resnet50(pretrained=True)
ct = 0
for child in model_ft.children():
ct += 1
if ct < 7:
    for param in child.parameters():
        param.requires_grad = False



lt=8
cntr=0

for child in model.children():
cntr+=1

if cntr < lt:
	print child
	for param in child.parameters():
		param.requires_grad = False
        num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)




def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
Then:

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()