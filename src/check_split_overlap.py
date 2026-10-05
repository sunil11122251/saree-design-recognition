import pandas as pd

METADATA_FILE = "results/metadata.csv"

df = pd.read_csv(METADATA_FILE)

# Use Kaggle dataset only
df = df[df["source"] == "kaggle"].copy()

# Correct group ID
df["group_id"] = (
    df["class"]
    + "__"
    + df["filename"].str.split(".rf.").str[0]
)

train_groups = set(
    df[df["split"] == "train"]["group_id"]
)

valid_groups = set(
    df[df["split"] == "valid"]["group_id"]
)

test_groups = set(
    df[df["split"] == "test"]["group_id"]
)

print("Train groups:", len(train_groups))
print("Validation groups:", len(valid_groups))
print("Test groups:", len(test_groups))

train_valid = train_groups & valid_groups
train_test = train_groups & test_groups
valid_test = valid_groups & test_groups

print("\nOverlap checks:")

print("Train ∩ Validation:", len(train_valid))
print("Train ∩ Test:", len(train_test))
print("Validation ∩ Test:", len(valid_test))

if len(train_valid) == 0 and len(train_test) == 0 and len(valid_test) == 0:
    print("\nSUCCESS: No group overlap between train, validation and test.")
else:
    print("\nWARNING: Group overlap detected.")