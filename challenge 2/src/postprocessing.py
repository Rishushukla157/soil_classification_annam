"""

Author: Annam.ai IIT Ropar
Team Name: OverTakers
Team Members: Member-1 rishu Shukla, member-2 Kaushik pal
Leaderboard Rank: 88

"""
submission = pd.DataFrame(predictions, columns=['image_id', 'label'])
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")

def postprocessing():
    print("This is the file for postprocessing")
  return 0
