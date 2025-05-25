"""

Author: Annam.ai IIT Ropar
Team Name: OverTakers
Team Members: Member-1 rishu Shukla, member-2 Kaushik pal
Leaderboard Rank: 55

"""
final_preds = []
for img_id in image_ids:
    # Stack fold predictions and average
    fold_probs = np.stack(model_preds[img_id], axis=0)
    avg_probs = np.mean(fold_probs, axis=0)
    pred_label = label_encoder.classes_[np.argmax(avg_probs)]
    final_preds.append(pred_label)
submission = pd.DataFrame({
    "image_id": image_ids,
    "soil_type": final_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

def postprocessing():
    print("This is the file for postprocessing")
  return 0
