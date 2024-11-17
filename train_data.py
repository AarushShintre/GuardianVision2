# Model Definition
class SimpleActionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleActionClassifier, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.sigmoid(base_model.fc.in_features, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Prepare dataset
dataset = SemiSupervisedDataset(annotated_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = SimpleActionClassifier(num_classes=len(set(action.split("_")[0] for action in annotated_data.keys())))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):  # Small epoch for demonstration
    for images, _, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()