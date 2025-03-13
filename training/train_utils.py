import torch
from tqdm import tqdm
import wandb

def evaluate(model, dataloader, class_weights, device, save_model=True):
    model.eval()
    total_loss = 0
    best_loss = float('inf')
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            mel = batch['mel'].to(device)
            text = batch['text'].to(device)
            target = batch['target'].to(device)
            output = model(mel, text)
            batch_loss = criterion(output, target)
            total_loss += batch_loss.item()
            progress_bar.set_postfix({"loss": batch_loss.item()})
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"avg_loss: {avg_loss}, accuracy: {accuracy}")
    if avg_loss < best_loss and save_model == True:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pt')
    
    model.train()
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, optimizer, class_weights, 
                device, number_epochs=1, run_name="accent-training"):
    if device == "cuda":
        wandb.init(project="finetune-contrastive-scottish", name=run_name)
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(number_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{number_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            mel = batch['mel'].to(device)
            text = batch['text'].to(device)
            target = batch['target'].to(device)
            output = model(mel, text)
            batch_loss = criterion(output, target)

            progress_bar.set_postfix({"loss": batch_loss.item()})
            
            batch_loss.backward()
            optimizer.step()
            if device == "cuda":
                wandb.log({"batch loss": batch_loss.item()})
            total_loss += batch_loss.item()
            
        val_loss, val_acc = evaluate(model, test_loader, class_weights, device)
        if device == "cuda":
            wandb.log({"val loss": val_loss, "epoch": epoch, "val acc": val_acc})
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")
    
    return model