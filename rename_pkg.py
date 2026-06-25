from pathlib import Path
root=Path(r"C:\Fake_Real_Major\Fake_Real_Major\Project_Major_food")
old=root/'webapp'/'ai_product_verification_system'
new=root/'webapp'/'ai_product_verification_system'
print('old exists', old.exists(), 'new exists', new.exists())
if old.exists() and not new.exists():
    old.rename(new)
    print('renamed')
else:
    print('rename skipped')
