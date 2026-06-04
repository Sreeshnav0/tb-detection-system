import os

def get_filenames(split):
    filenames = set()
    for label in ['Normal', 'Tuberculosis']:
        path = os.path.join('dataset', split, label)
        if os.path.exists(path):
            filenames.update(os.listdir(path))
    return filenames

def main():
    train_files = get_filenames('train')
    val_files = get_filenames('val')
    test_files = get_filenames('test')

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    leakage_train_val = train_files.intersection(val_files)
    leakage_train_test = train_files.intersection(test_files)
    leakage_val_test = val_files.intersection(test_files)

    found_leakage = False

    if leakage_train_val:
        print(f"\nData leakage detected between Train and Val: {len(leakage_train_val)} files")
        print(list(leakage_train_val)[:10])
        found_leakage = True

    if leakage_train_test:
        print(f"\nData leakage detected between Train and Test: {len(leakage_train_test)} files")
        print(list(leakage_train_test)[:10])
        found_leakage = True

    if leakage_val_test:
        print(f"\nData leakage detected between Val and Test: {len(leakage_val_test)} files")
        print(list(leakage_val_test)[:10])
        found_leakage = True

    if not found_leakage:
        print("\nNo data leakage detected.")

if __name__ == "__main__":
    main()
