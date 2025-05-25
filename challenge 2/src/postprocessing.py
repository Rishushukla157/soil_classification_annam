"""

Author: Annam.ai IIT Ropar
Team Name: OverTakers
Team Members: Member-1 rishu Shukla, member-2 Kaushik pal
Leaderboard Rank: 88

"""
test_ids = pd.read_csv("/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv")

# Test Dataset
class TestDataset(Dataset):
    def _init_(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def _len_(self):
        return len(self.ids)

    def _getitem_(self, idx):
        img_name = self.ids.loc[idx, 'image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestDataset(test_ids, test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predict
model.eval()
predictions = []

with torch.no_grad():
    for images, image_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().cpu().numpy().flatten()

        for img_name, pred in zip(image_names, preds):
            predictions.append((img_name, pred))
submission = pd.DataFrame(predictions, columns=['image_id', 'label'])
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")

def postprocessing():
    print("This is the file for postprocessing")
  return 0
