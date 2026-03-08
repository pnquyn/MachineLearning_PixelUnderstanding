import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')

languages = df['language'].unique()
print(f"Tìm thấy {len(languages)} ngôn ngữ: {', '.join(languages)}")

train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['language']  
)

train_df.to_csv('data/train_split.csv', index=False)
val_df.to_csv('data/val_split.csv', index=False)

print("\n--- Thống kê sau khi chia ---")
stats = []
for lang in languages:
    total = len(df[df['language'] == lang])
    train_count = len(train_df[train_df['language'] == lang])
    val_count = len(val_df[val_df['language'] == lang])
    stats.append({
        'Language': lang,
        'Total': total,
        'Train': f"{train_count} ({train_count/total:.1%})",
        'Val': f"{val_count} ({val_count/total:.1%})"
    })

print(pd.DataFrame(stats).to_string(index=False))
print(f"\nTổng Train: {len(train_df)} mẫu")
print(f"Tổng Val: {len(val_df)} mẫu")