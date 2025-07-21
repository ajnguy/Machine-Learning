import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import random
from sklearn.model_selection import train_test_split

start_time = time.time()

class MulticlassClassfier(nn.Module):
    def __init__(self, input_size, num_classes, optimizer, learning_rate, num_iterations, convergence_tol):
        super(MulticlassClassfier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.num_iterations = num_iterations
        self.convergence_tol = convergence_tol

    def discriminant(self, x):
        return self.linear(x)

    def loss(self, y, g):
        N = g.shape[0]
        loss = -torch.gather(g,1,y.long()) + torch.logsumexp(g,axis=1,keepdim=True)
        return loss

    def fit(self, x_train, y_train, x_val, y_val, batch_size=64, patience=50):
        x_train = x_train.view(x_train.size(0), -1)
        x_val = x_val.view(x_val.size(0), -1)
        best_val_loss = float('inf')
        consecutive_no_improvement = 0

        for i in range(self.num_iterations):
            indices = list(range(len(x_train)))
            random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, len(x_train), batch_size):
                end = start + batch_size
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                y = self.discriminant(x_batch)
                loss = torch.mean(self.loss(y_batch, y))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if i % 10 == 0:
                val_loss = self.validate(x_val, y_val)
                print(f"Epoch {i + 1}/{self.num_iterations}, Training Loss: {loss}, Validation Loss: {val_loss}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    consecutive_no_improvement = 0
                else:
                    consecutive_no_improvement += 1

                if consecutive_no_improvement >= patience:
                    print(f"No improvement for {patience} epochs. Stopping training.")
                    break
                if loss < self.convergence_tol:
                    break

    def validate(self, x_val, y_val):
        with torch.no_grad():
            y = self.discriminant(x_val)
            loss = torch.mean(self.loss(y_val, y))
            return loss

    def predict(self, x):
        x = x.view(x.size(0), -1)
        y = self.discriminant(x)
        y_predict = torch.argmax(y, dim=1)
        y_predict = y_predict.numpy()
        return y_predict

# TRAIN
def load_data(root_dir):
    images = []
    labels = []
    classes = os.listdir(root_dir)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith(".jpg"):
                file_path = os.path.join(class_dir, file)
                img = Image.open(file_path)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(class_to_idx[class_name])

    return np.array(images), np.array(labels)

root_dir = "./data/train"
images, labels = load_data(root_dir)

x_train = torch.tensor(images, dtype=torch.float32)
y_train = torch.tensor(labels, dtype=torch.long).reshape((labels.size, 1)) - 1

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

x_train = (x_train - x_train.mean()) / x_train.std()
x_val = (x_val - x_val.mean()) / x_val.std()


input_size = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
num_classes = len(torch.unique(y_train))

model = MulticlassClassfier(input_size=input_size, 
                            num_classes=num_classes, 
                            optimizer=optim.Adam, 
                            learning_rate=0.001, 
                            num_iterations=5000, 
                            convergence_tol=1e-4)
model.fit(x_train, y_train, x_val, y_val)
y_train_pred = model.predict(x_train)

num_errors = 0
for predicted_label, true_label in zip(y_train_pred, y_train.numpy().reshape(-1)):
    if predicted_label != true_label:
        num_errors += 1
train_error = num_errors / len(y_train)
print(f"Train Error Rate: {train_error * 100}%")

# TEST
model.eval()
test_dir = "./data/test"
images_test = []
for file_name in os.listdir(test_dir):
    if file_name.endswith(".jpg"):
        file_path = os.path.join(test_dir, file_name)
        img = Image.open(file_path)
        img_array = np.array(img) / 255.0
        images_test.append(img_array)
images_test = np.array(images_test)
x_test = torch.tensor(images_test, dtype=torch.float32)
x_test = (x_test - x_test.mean()) / x_test.std()
y_test_pred = model.predict(x_test)

class_names = os.listdir("./data/train")[1:]
idx_to_class = {i: cls for i, cls in enumerate(class_names)}
y_test_pred_classes = [idx_to_class[idx] for idx in y_test_pred]

def write_prediction_file(predictions,instance_filenames,output_filename):
    with open(output_filename,'w') as f:
        f.write("Instance, Label\n")
        for i,name in enumerate(instance_filenames):
            f.write(f"{name}, {predictions[i]}\n")

dir = "./data/test"
file_names=[]
for f in os.listdir(dir):
    if not ".jpg" in f: continue
    file_names.append(f)

write_prediction_file(y_test_pred_classes,file_names,"linear_predictions.csv")

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Total runtime: {elapsed_time} seconds")