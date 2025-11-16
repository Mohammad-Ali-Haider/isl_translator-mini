from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_size=42, num_classes=9, dropout_rate=0.3):
        """
        Args:
            input_size: Number of input features (42 for x,y only, 63 for x,y,z)
            num_classes: Number of output classes (9 for digits 1-9, 10 for 0-9)
            dropout_rate: Dropout probability
        """
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x