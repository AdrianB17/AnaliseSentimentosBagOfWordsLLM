# AnaliseSentimentosBagOfWordsLLM

## 1. Execution in 1 to 2 seconds per epoch with GPU T4 or 15 seconds per epoch on CPU.
### ❌ Before:
```python
def __getitem__(self, idx):
        sample = self.data[idx]
        target = sample["label"]
        line = sample["text"]
        target = 1 if target == 1 else 0

        # one-hot encoding
        X = torch.zeros(len(self.vocab) + 1)
        for word in encode_sentence(line, self.vocab):
            X[word] = 1
````
Problems: 
* In each call to __getitem__, preprocessing, encoding, and creation of the one-hot vector is performed.
* This repeats unnecessary work every time a sample is accessed.

### ✅ After:
```python
def __getitem__(self, idx):
        return self.encoded_data[idx]
````
Improvement:
* All texts are preprocessed and encoded only once in the constructor (__init__)
* self.encoded_data is stored, and __getitem__ only accesses this list.
📌 Result: Increases training efficiency (faster, less CPU).


## 2. Improvement of the Tokenizer

### ❌ Antes:
```python
for sample in train_dataset:
    counter.update(sample["text"].split())
````
Problems:
* Tokenization is done simply with text.split(), without preprocessing the text. 
* This includes uppercase letters, punctuation marks, etc.

### ✅ Despues:
```python
def preprocess_text(text):
    # Converts to lowercase
    text = text.lower()
    # Removes punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
````
Improvement:
* A preprocess_text(text) function is included that converts the text to lowercase and removes punctuation

📌 Result: Improves vocabulary quality and reduces the number of out-of-vocabulary tokens (OOV).


## 3. Mejorar LR (tasa de aprendizaje) para lograr mejor precisión en el conjunto de prueba.
Tests were conducted with different values of the learning rate while keeping the other parameters constant (SGD optimization, number of epochs, batch size, loss function). 
The learning rate was selected that allowed for the best accuracy on the validation set (test), without causing instability in the loss or loss of convergence. 
For LR= 0.01: 
![dos ceros](https://github.com/user-attachments/assets/a0764d08-be3f-42a0-9a88-5e2b09d3ee85)

For LR= 0.001:
![image](https://github.com/user-attachments/assets/355d6301-f184-41e0-930a-16a9e17ae466)


## 4. Conceptual error in the calculation of the loss incurred in each period.
### ❌ Antes:
```python
for inputs, targets in train_loader:
    ...
    loss = criterion(logits.squeeze(), targets.float())
    ...
print(f'Loss: {loss.item():.4f}')
````
Problem:
* At the end of each epoch, only the loss of the last batch is printed, not the average loss of the entire epoch. This does not reflect the overall performance of the model on all training data, but only on the last batch.

### ✅ Despues:
```python
total_epoch_loss += loss.item() * inputs.size(0)
...
average_epoch_loss = total_epoch_loss / len(train_loader.dataset)
print(f'Train Loss: {average_epoch_loss:.4f}')
````
Improvement:
* The weighted loss of each batch is accumulated.
* At the end of the epoch, it is divided by the total number of samples to obtain the average loss of the epoch.
* This average loss is printed, which accurately represents the model's performance during that epoch.

## 5. Include in the training loop (the epoch loop), also the validation loop, with the printing of the loss (training and validation). Use the corrected version in the calculation of the loss per epoch. Split the training set into training and validation.
### ✅ Despues:
```python
# Split into train and validation
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])
...

# Function to calculate loss and accuracy
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.squeeze(), targets.float())
            total_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(logits) >= 0.5).float()
            correct_predictions += (predicted.squeeze() == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy
...

print(f'Epoch [{epoch+1}/{num_epochs}], '
  f'Train Loss: {average_epoch_loss:.4f}, '
  f'Validation Loss: {val_loss:.4f}, '
  f'Validation Accuracy: {val_accuracy:.4f}, '
  f'Elapsed Time: {epoch_duration:.2f} sec')
````
Benefits:
* Early detection of overfitting: By comparing the loss and accuracy of the validation set with those of the training set, it is possible to identify when the model starts to degrade its performance on new data.
* Hyperparameter optimization: Validation allows for the evaluation of the impact of changes in hyperparameters (such as learning rate).
