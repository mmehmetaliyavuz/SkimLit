import torch
from tqdm import tqdm
def train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, accuracy_fn, device, epochs=10):
    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")
        print('-' * 10)
        
        train_loss, train_acc = 0, 0
        model.to(device)
        model.train()
        for batch, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_pred_train = model(X_train)
            loss_train = loss_fn(y_pred_train, y_train)
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item()
            train_acc += accuracy_fn(y_true=y_train, y_pred=y_pred_train.argmax(dim=1))
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_pred_test = model(X_test)
                loss_test = loss_fn(y_pred_test, y_test)
                test_loss += loss_test.item()
                test_acc += accuracy_fn(y_true=y_test, y_pred=y_pred_test.argmax(dim=1))
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
    print("Training complete")
    return results
