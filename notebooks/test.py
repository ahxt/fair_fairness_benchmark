# %%
import datasets
from datasets import load_dataset


# %%
# dataset = load_dataset("ahxt/adult", "income", use_auth_token=True) 

# dataset = load_dataset("huggan/CelebA-faces-with-attributes") 
# dataset = load_dataset("nlphuji/utk_faces") 
dataset = load_dataset("affahrizain/jigsaw-toxic-comment") 




# %%
# dataset.push_to_hub("ahxt/adult_test2", private=True)

# %%
print( dataset )

# %%
dataset

# %%
import pandas as pd

dataset["train"].features

# %%



